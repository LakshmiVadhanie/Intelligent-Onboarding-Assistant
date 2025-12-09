import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalSensitivityAnalyzer:
    
    def __init__(self, retriever, evaluator):

        self.retriever = retriever
        self.evaluator = evaluator
        logger.info(" Sensitivity analyzer initialized")
    
    def analyze_query_length_sensitivity(self, test_queries: List[Dict]) -> Dict:

        logger.info("\n" + "="*80)
        logger.info(" ANALYZING QUERY LENGTH SENSITIVITY")
        logger.info("="*80)
        
        length_groups = defaultdict(list)
        
        for query_data in test_queries:
            query = query_data['query']
            word_count = len(query.split())
            
            if word_count <= 3:
                group = 'short (1-3 words)'
            elif word_count <= 6:
                group = 'medium (4-6 words)'
            else:
                group = 'long (7+ words)'
            
            length_groups[group].append(query_data)
        
        results = {}
        
        for group_name, queries in length_groups.items():
            if not queries:
                continue
            
            logger.info(f"\n📊 Evaluating {group_name}: {len(queries)} queries")
            
            eval_results = self.evaluator.evaluate(queries, k_values=[5])
            
            results[group_name] = {
                'num_queries': len(queries),
                'precision_at_5': eval_results['metrics_by_k']['k=5']['precision'],
                'recall_at_5': eval_results['metrics_by_k']['k=5']['recall'],
                'mrr': eval_results['mrr']
            }
            
            logger.info(f"  Precision@5: {results[group_name]['precision_at_5']:.4f}")
            logger.info(f"  MRR: {results[group_name]['mrr']:.4f}")
        
        return results
    
    def analyze_source_type_sensitivity(self, test_queries: List[Dict]) -> Dict:

        logger.info("\n" + "="*80)
        logger.info(" ANALYZING SOURCE TYPE SENSITIVITY")
        logger.info("="*80)
        
        source_performance = defaultdict(lambda: {
            'retrieved_count': 0,
            'relevant_count': 0,
            'ranks': []
        })
        
        for query_data in test_queries:
            query = query_data['query']
            relevant_ids = query_data.get('relevant_ids', [])
            
            results = self.retriever.retrieve(query, k=10)
            
            for rank, doc in enumerate(results, 1):
                source_type = doc['metadata'].get('source_type', 
                             doc['metadata'].get('source', 'unknown'))
                
                source_performance[source_type]['retrieved_count'] += 1
                
                if doc['id'] in relevant_ids:
                    source_performance[source_type]['relevant_count'] += 1
                    source_performance[source_type]['ranks'].append(rank)
        
        results = {}
        
        for source_type, stats in source_performance.items():
            avg_rank = np.mean(stats['ranks']) if stats['ranks'] else 0
            
            results[source_type] = {
                'retrieved_count': stats['retrieved_count'],
                'relevant_retrieved': stats['relevant_count'],
                'avg_rank_when_relevant': avg_rank,
                'relevance_rate': (stats['relevant_count'] / stats['retrieved_count'] 
                                  if stats['retrieved_count'] > 0 else 0)
            }
            
            logger.info(f"\n📄 {source_type}:")
            logger.info(f"  Retrieved: {stats['retrieved_count']} times")
            logger.info(f"  Relevant: {stats['relevant_count']} times")
            logger.info(f"  Relevance rate: {results[source_type]['relevance_rate']:.2%}")
        
        return results
    
    def analyze_embedding_dimension_impact(self) -> Dict:

        logger.info("\n" + "="*80)
        logger.info(" ANALYZING EMBEDDING DIMENSION IMPACT")
        logger.info("="*80)
        
        model_info_path = Path("models/embeddings/model_info.json")
        
        if model_info_path.exists():
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            embedding_dim = model_info['embedding_dim']
            num_embeddings = model_info['num_embeddings']
            
            total_params = embedding_dim * num_embeddings
            memory_mb = (total_params * 4) / (1024 * 1024) 
            
            results = {
                'embedding_dimension': embedding_dim,
                'num_documents': num_embeddings,
                'total_parameters': total_params,
                'memory_mb': memory_mb,
                'model_name': model_info['model_name'],
                'analysis': {
                    'dimension_efficiency': 'optimal' if embedding_dim < 512 else 'high',
                    'memory_footprint': 'low' if memory_mb < 100 else 'moderate',
                    'inference_speed': 'fast' if embedding_dim < 512 else 'moderate'
                }
            }
            
            logger.info(f"\n Embedding Analysis:")
            logger.info(f"  Dimensions: {embedding_dim}")
            logger.info(f"  Documents: {num_embeddings}")
            logger.info(f"  Memory: {memory_mb:.2f} MB")
            logger.info(f"  Efficiency: {results['analysis']['dimension_efficiency']}")
            
            return results
        else:
            logger.warning("Model info not found")
            return {}
    
    def analyze_retrieval_k_sensitivity(self, test_queries: List[Dict]) -> Dict:

        logger.info("\n" + "="*80)
        logger.info(" ANALYZING K-VALUE SENSITIVITY")
        logger.info("="*80)
        
        k_values = [1, 3, 5, 10, 20]
        results = {}
        
        for k in k_values:
            logger.info(f"\n Evaluating K={k}...")
            
            eval_results = self.evaluator.evaluate(test_queries, k_values=[k])
            
            results[f'k={k}'] = {
                'precision': eval_results['metrics_by_k'][f'k={k}']['precision'],
                'recall': eval_results['metrics_by_k'][f'k={k}']['recall'],
                'f1': eval_results['metrics_by_k'][f'k={k}']['f1']
            }
            
            logger.info(f"  Precision: {results[f'k={k}']['precision']:.4f}")
            logger.info(f"  Recall: {results[f'k={k}']['recall']:.4f}")
        
        return results
    
    def generate_sensitivity_report(self, test_queries: List[Dict]) -> Dict:

        logger.info("\n" + "="*80)
        logger.info(" GENERATING COMPREHENSIVE SENSITIVITY REPORT")
        logger.info("="*80)
        
        report = {
            'query_length_sensitivity': self.analyze_query_length_sensitivity(test_queries),
            'source_type_sensitivity': self.analyze_source_type_sensitivity(test_queries),
            'embedding_analysis': self.analyze_embedding_dimension_impact(),
            'k_value_sensitivity': self.analyze_retrieval_k_sensitivity(test_queries)
        }
        
        report['insights'] = self._generate_insights(report)
        
        return report
    
    def _generate_insights(self, report: Dict) -> List[str]:
        insights = []
        
        if 'query_length_sensitivity' in report:
            lengths = report['query_length_sensitivity']
            if lengths:
                best_length = max(lengths.items(), 
                                 key=lambda x: x[1].get('precision_at_5', 0))
                insights.append(
                    f"Best performance with {best_length[0]} queries "
                    f"(Precision@5: {best_length[1]['precision_at_5']:.2%})"
                )
        
        if 'k_value_sensitivity' in report:
            k_results = report['k_value_sensitivity']
            if k_results:
                best_k = max(k_results.items(), 
                           key=lambda x: x[1].get('f1', 0))
                insights.append(
                    f"Optimal K value: {best_k[0]} (F1: {best_k[1]['f1']:.2%})"
                )
        
        if 'embedding_analysis' in report:
            emb = report['embedding_analysis']
            if emb:
                insights.append(
                    f"Embedding efficiency: {emb['analysis']['dimension_efficiency']} "
                    f"({emb['embedding_dimension']} dims, {emb['memory_mb']:.1f}MB)"
                )
        
        return insights
    
    def save_report(self, report: Dict, output_path: str = "experiments/sensitivity_analysis.json"):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n Sensitivity report saved: {output_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from retrieval.retriever import BaselineRetriever
    from evaluation.metrics import RetrievalEvaluator
    
    print("\n" + "="*80)
    print(" SENSITIVITY ANALYSIS")
    print("="*80)
    
    retriever = BaselineRetriever()
    evaluator = RetrievalEvaluator(retriever)
    analyzer = RetrievalSensitivityAnalyzer(retriever, evaluator)
    
    test_queries = [
        {'query': 'sustainability', 'relevant_ids': ['doc_8']},
        {'query': 'What is GitLab approach to sustainability?', 'relevant_ids': ['doc_8']},
        {'query': 'risk management', 'relevant_ids': ['doc_12']},
        {'query': 'How does risk management work at GitLab?', 'relevant_ids': ['doc_12']},
        {'query': 'CI/CD', 'relevant_ids': ['doc_3']},
        {'query': 'Tell me about the CI/CD UX meeting discussions', 'relevant_ids': ['doc_3']},
    ]
    
    report = analyzer.generate_sensitivity_report(test_queries)
    
    print("\n" + "="*80)
    print(" KEY INSIGHTS")
    print("="*80)
    
    for i, insight in enumerate(report['insights'], 1):
        print(f"{i}. {insight}")
    
    analyzer.save_report(report)
    
    print("\n" + "="*80)
    print(" SENSITIVITY ANALYSIS COMPLETE!")
    print("="*80)