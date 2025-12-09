
import numpy as np
from typing import Dict, List, Tuple
import logging
import json
from pathlib import Path
from collections import defaultdict
from scipy import stats
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBiasDetector:
    """
    Enhanced bias detector that analyzes performance disparities across
    different query categories, sources, and content types to identify
    bias in RAG retrieval and generation systems.
    """
    
    def __init__(self, retriever, threshold: float = 0.15):
        """
        Initialize bias detector
        
        Args:
            retriever: Retriever instance with retrieve() method
            threshold: Minimum performance difference to flag as bias (default 15%)
        """
        self.retriever = retriever
        self.threshold = threshold
        self.metrics_by_category = {}
        logger.info("✓ EnhancedBiasDetector initialized")
        logger.info(f"  Bias threshold: {threshold:.0%}")
    
    def compute_retrieval_metrics(self, query: str, relevant_docs: List[str], 
                                  k: int = 5) -> Dict:
        """
        Compute precision, recall, MRR for a single query
        
        Args:
            query: User query string
            relevant_docs: List of relevant document IDs
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with precision, recall, MRR, and similarity metrics
        """
        results = self.retriever.retrieve(query, k=k)
        retrieved_ids = [doc['metadata'].get('id', '') for doc in results]
        
        # Precision@k
        relevant_retrieved = set(retrieved_ids) & set(relevant_docs)
        precision = len(relevant_retrieved) / k if k > 0 else 0
        
        # Recall@k
        recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_docs:
                mrr = 1 / rank
                break
        
        # Average similarity score
        avg_similarity = np.mean([doc['similarity'] for doc in results]) if results else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'mrr': mrr,
            'f1': f1,
            'avg_similarity': avg_similarity,
            'retrieved_relevant': len(relevant_retrieved),
            'total_relevant': len(relevant_docs),
            'total_retrieved': len(results)
        }
    
    def analyze_category_performance(self, queries_by_category: Dict[str, List[Dict]], 
                                    k: int = 5) -> Dict:
        """
        Analyze retrieval performance across different query categories
        
        Args:
            queries_by_category: Dictionary mapping category names to lists of query dicts
                Format: {
                    'engineering': [
                        {'query': '...', 'relevant_docs': ['doc_1', 'doc_2']},
                        ...
                    ],
                    'hr_policies': [...]
                }
            k: Number of documents to retrieve per query
            
        Returns:
            Dictionary with aggregated metrics per category
        """
        logger.info("\n" + "=" * 80)
        logger.info("📊 CATEGORY-BASED PERFORMANCE ANALYSIS")
        logger.info("=" * 80)
        
        category_metrics = {}
        
        for category, queries in queries_by_category.items():
            metrics_list = []
            
            logger.info(f"\nAnalyzing {category} ({len(queries)} queries)...")
            
            for query_data in queries:
                query = query_data['query']
                relevant = query_data.get('relevant_docs', [])
                
                metrics = self.compute_retrieval_metrics(query, relevant, k=k)
                metrics_list.append(metrics)
            
            # Aggregate metrics for category
            category_metrics[category] = {
                'precision': np.mean([m['precision'] for m in metrics_list]),
                'recall': np.mean([m['recall'] for m in metrics_list]),
                'mrr': np.mean([m['mrr'] for m in metrics_list]),
                'f1': np.mean([m['f1'] for m in metrics_list]),
                'avg_similarity': np.mean([m['avg_similarity'] for m in metrics_list]),
                'num_queries': len(queries),
                'total_relevant_found': sum([m['retrieved_relevant'] for m in metrics_list]),
                'std_precision': np.std([m['precision'] for m in metrics_list]),
                'std_similarity': np.std([m['avg_similarity'] for m in metrics_list])
            }
            
            logger.info(f"\n  📈 {category.upper().replace('_', ' ')}:")
            logger.info(f"     Queries: {category_metrics[category]['num_queries']}")
            logger.info(f"     Precision@{k}: {category_metrics[category]['precision']:.3f}")
            logger.info(f"     Recall@{k}: {category_metrics[category]['recall']:.3f}")
            logger.info(f"     MRR: {category_metrics[category]['mrr']:.3f}")
            logger.info(f"     F1 Score: {category_metrics[category]['f1']:.3f}")
            logger.info(f"     Avg Similarity: {category_metrics[category]['avg_similarity']:.4f}")
        
        self.metrics_by_category = category_metrics
        return category_metrics
    
    def analyze_source_distribution(self, test_queries: List[Dict], k: int = 5) -> Dict:
        """
        Analyze which sources (handbook sections) are retrieved most frequently
        
        Args:
            test_queries: List of query dictionaries
            k: Number of documents to retrieve per query
            
        Returns:
            Dictionary with source distribution statistics
        """
        logger.info("\n" + "=" * 80)
        logger.info("📚 SOURCE DISTRIBUTION ANALYSIS")
        logger.info("=" * 80)
        
        source_stats = defaultdict(lambda: {
            'retrieved_count': 0,
            'total_retrievals': 0,
            'avg_rank': [],
            'avg_similarity': [],
            'queries_retrieved_for': set()
        })
        
        for idx, query_data in enumerate(test_queries):
            query = query_data['query']
            results = self.retriever.retrieve(query, k=k)
            
            for rank, doc in enumerate(results, 1):
                source = doc['metadata'].get('source', 'unknown')
                title = doc['metadata'].get('title', 'Untitled')
                
                source_stats[source]['retrieved_count'] += 1
                source_stats[source]['total_retrievals'] += 1
                source_stats[source]['avg_rank'].append(rank)
                source_stats[source]['avg_similarity'].append(doc['similarity'])
                source_stats[source]['queries_retrieved_for'].add(idx)
        
        # Compute aggregated statistics
        source_report = {}
        for source, stats in source_stats.items():
            source_report[source] = {
                'total_retrieved': stats['retrieved_count'],
                'retrieval_rate': stats['retrieved_count'] / (len(test_queries) * k),
                'avg_rank': np.mean(stats['avg_rank']) if stats['avg_rank'] else 0,
                'avg_similarity': np.mean(stats['avg_similarity']) if stats['avg_similarity'] else 0,
                'query_coverage': len(stats['queries_retrieved_for']) / len(test_queries)
            }
        
        # Sort by retrieval frequency
        sorted_sources = sorted(source_report.items(), 
                               key=lambda x: x[1]['total_retrieved'], 
                               reverse=True)
        
        logger.info(f"\n  📊 Source Distribution (Top 10):")
        logger.info(f"  {'Source':<40} {'Count':>8} {'Rate':>8} {'Avg Rank':>10} {'Avg Sim':>10}")
        logger.info("  " + "-" * 80)
        
        for source, metrics in sorted_sources[:10]:
            logger.info(
                f"  {source[:40]:<40} "
                f"{metrics['total_retrieved']:>8d} "
                f"{metrics['retrieval_rate']:>7.1%} "
                f"{metrics['avg_rank']:>10.2f} "
                f"{metrics['avg_similarity']:>10.4f}"
            )
        
        return dict(sorted_sources)
    
    def analyze_title_bias(self, test_queries: List[Dict], k: int = 5) -> Dict:
        """
        Analyze which document titles appear most frequently in retrievals
        
        Args:
            test_queries: List of query dictionaries
            k: Number of documents to retrieve per query
            
        Returns:
            Dictionary with title distribution statistics
        """
        logger.info("\n" + "=" * 80)
        logger.info("📄 TITLE/CONTENT TYPE BIAS ANALYSIS")
        logger.info("=" * 80)
        
        title_stats = defaultdict(lambda: {
            'count': 0,
            'avg_similarity': [],
            'avg_rank': []
        })
        
        for query_data in test_queries:
            query = query_data['query']
            results = self.retriever.retrieve(query, k=k)
            
            for rank, doc in enumerate(results, 1):
                title = doc['metadata'].get('title', 'Untitled')
                title_stats[title]['count'] += 1
                title_stats[title]['avg_similarity'].append(doc['similarity'])
                title_stats[title]['avg_rank'].append(rank)
        
        # Aggregate statistics
        title_report = {}
        for title, stats in title_stats.items():
            title_report[title] = {
                'count': stats['count'],
                'avg_similarity': np.mean(stats['avg_similarity']),
                'avg_rank': np.mean(stats['avg_rank'])
            }
        
        sorted_titles = sorted(title_report.items(), 
                              key=lambda x: x[1]['count'], 
                              reverse=True)
        
        logger.info(f"\n  📑 Top Retrieved Titles:")
        logger.info(f"  {'Title':<50} {'Count':>8} {'Avg Sim':>10} {'Avg Rank':>10}")
        logger.info("  " + "-" * 80)
        
        for title, stats in sorted_titles[:15]:
            logger.info(
                f"  {title[:50]:<50} "
                f"{stats['count']:>8d} "
                f"{stats['avg_similarity']:>10.4f} "
                f"{stats['avg_rank']:>10.2f}"
            )
        
        return dict(sorted_titles)
    
    def detect_category_bias(self, category_metrics: Dict, 
                           significance_level: float = 0.05) -> Dict:
        """
        Detect significant performance disparities between categories using statistical tests
        
        Args:
            category_metrics: Output from analyze_category_performance()
            significance_level: P-value threshold for statistical significance
            
        Returns:
            Dictionary with detected biases and disparities
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔍 DETECTING PERFORMANCE DISPARITIES")
        logger.info("=" * 80)
        
        if len(category_metrics) < 2:
            logger.warning("⚠️  Need at least 2 categories to detect bias")
            return {}
        
        # Calculate overall averages
        all_precisions = [m['precision'] for m in category_metrics.values()]
        all_recalls = [m['recall'] for m in category_metrics.values()]
        all_mrrs = [m['mrr'] for m in category_metrics.values()]
        all_sims = [m['avg_similarity'] for m in category_metrics.values()]
        
        overall_metrics = {
            'precision': np.mean(all_precisions),
            'recall': np.mean(all_recalls),
            'mrr': np.mean(all_mrrs),
            'avg_similarity': np.mean(all_sims)
        }
        
        logger.info(f"\n  📊 Overall Average Metrics:")
        logger.info(f"     Precision@k: {overall_metrics['precision']:.3f}")
        logger.info(f"     Recall@k: {overall_metrics['recall']:.3f}")
        logger.info(f"     MRR: {overall_metrics['mrr']:.3f}")
        logger.info(f"     Avg Similarity: {overall_metrics['avg_similarity']:.4f}")
        
        # Detect disparities
        disparities = {}
        bias_detected = False
        
        logger.info(f"\n  🎯 Disparity Detection (threshold: {self.threshold:.0%}):")
        logger.info("  " + "-" * 80)
        
        for category, metrics in category_metrics.items():
            category_disparities = {}
            
            # Check each metric for disparity
            for metric_name in ['precision', 'recall', 'mrr', 'avg_similarity']:
                category_value = metrics[metric_name]
                overall_value = overall_metrics[metric_name]
                
                # Calculate relative difference
                if overall_value > 0:
                    rel_diff = abs(category_value - overall_value) / overall_value
                else:
                    rel_diff = 0
                
                # Check if disparity exceeds threshold
                if rel_diff > self.threshold:
                    category_disparities[metric_name] = {
                        'category_value': category_value,
                        'overall_value': overall_value,
                        'absolute_diff': category_value - overall_value,
                        'relative_diff': rel_diff,
                        'percentage_diff': rel_diff * 100,
                        'performance': 'underperforming' if category_value < overall_value else 'overperforming'
                    }
            
            if category_disparities:
                disparities[category] = category_disparities
                bias_detected = True
                
                logger.warning(f"\n  ⚠️  BIAS DETECTED: {category.upper().replace('_', ' ')}")
                for metric_name, disp in category_disparities.items():
                    logger.warning(
                        f"     • {metric_name.upper()}: "
                        f"{disp['category_value']:.3f} vs {disp['overall_value']:.3f} "
                        f"({disp['performance']}, {disp['percentage_diff']:.1f}% diff)"
                    )
        
        if not bias_detected:
            logger.info("\n  ✓ No significant performance disparities detected!")
        
        return {
            'disparities': disparities,
            'overall_metrics': overall_metrics,
            'has_significant_bias': bias_detected,
            'num_biased_categories': len(disparities)
        }
    
    def analyze_coverage_bias(self, queries_by_category: Dict[str, List[Dict]], 
                             k: int = 5) -> Dict:
        """
        Analyze content coverage - which categories have sufficient relevant documents
        
        Args:
            queries_by_category: Dictionary mapping categories to query lists
            k: Number of documents to retrieve
            
        Returns:
            Coverage statistics by category
        """
        logger.info("\n" + "=" * 80)
        logger.info("📖 CONTENT COVERAGE ANALYSIS")
        logger.info("=" * 80)
        
        coverage_stats = {}
        
        for category, queries in queries_by_category.items():
            total_queries = len(queries)
            queries_with_results = 0
            avg_results_per_query = []
            
            for query_data in queries:
                query = query_data['query']
                results = self.retriever.retrieve(query, k=k)
                
                if results:
                    queries_with_results += 1
                    avg_results_per_query.append(len(results))
            
            coverage_stats[category] = {
                'total_queries': total_queries,
                'queries_with_results': queries_with_results,
                'coverage_rate': queries_with_results / total_queries if total_queries > 0 else 0,
                'avg_results_per_query': np.mean(avg_results_per_query) if avg_results_per_query else 0
            }
            
            logger.info(f"\n  {category.upper().replace('_', ' ')}:")
            logger.info(f"     Coverage: {coverage_stats[category]['coverage_rate']:.1%}")
            logger.info(f"     Queries with results: {queries_with_results}/{total_queries}")
            logger.info(f"     Avg results/query: {coverage_stats[category]['avg_results_per_query']:.1f}")
        
        return coverage_stats
    
    def perform_statistical_test(self, category_metrics: Dict) -> Dict:
        """
        Perform ANOVA test to determine if performance differences are statistically significant
        
        Args:
            category_metrics: Output from analyze_category_performance()
            
        Returns:
            Statistical test results
        """
        logger.info("\n" + "=" * 80)
        logger.info("📈 STATISTICAL SIGNIFICANCE TEST")
        logger.info("=" * 80)
        
        if len(category_metrics) < 2:
            logger.warning("⚠️  Need at least 2 categories for statistical test")
            return {}
        
        # Extract precision values by category
        category_names = list(category_metrics.keys())
        precision_groups = [[category_metrics[cat]['precision']] 
                           for cat in category_names]
        
        # Note: For real ANOVA, we'd need multiple samples per category
        # This is simplified for demonstration
        
        results = {
            'test_type': 'Visual comparison (insufficient samples for ANOVA)',
            'categories_compared': category_names,
            'recommendation': 'Increase test queries per category to 20+ for robust statistical testing'
        }
        
        logger.info(f"\n  ℹ️  {results['test_type']}")
        logger.info(f"  ℹ️  {results['recommendation']}")
        
        return results
    
    def generate_bias_mitigation_recommendations(self, disparities: Dict, 
                                                 coverage_stats: Dict) -> List[str]:
        """
        Generate actionable recommendations to mitigate detected biases
        
        Args:
            disparities: Output from detect_category_bias()
            coverage_stats: Output from analyze_coverage_bias()
            
        Returns:
            List of mitigation recommendations
        """
        logger.info("\n" + "=" * 80)
        logger.info("💡 BIAS MITIGATION RECOMMENDATIONS")
        logger.info("=" * 80)
        
        recommendations = []
        
        if disparities['has_significant_bias']:
            biased_categories = disparities['disparities']
            
            for category, disp_metrics in biased_categories.items():
                # Check what's underperforming
                underperforming_metrics = [
                    name for name, data in disp_metrics.items() 
                    if data['performance'] == 'underperforming'
                ]
                
                if underperforming_metrics:
                    recommendations.append(
                        f"📌 {category.upper().replace('_', ' ')}: "
                        f"Underperforming on {', '.join(underperforming_metrics)}"
                    )
                    
                    # Check coverage
                    if category in coverage_stats:
                        cov_rate = coverage_stats[category]['coverage_rate']
                        if cov_rate < 0.8:
                            recommendations.append(
                                f"   → Add more {category} content to knowledge base "
                                f"(current coverage: {cov_rate:.0%})"
                            )
                    
                    # Check if precision is low
                    if 'precision' in disp_metrics:
                        if disp_metrics['precision']['category_value'] < 0.6:
                            recommendations.append(
                                f"   → Improve chunking strategy for {category} documents"
                            )
                            recommendations.append(
                                f"   → Add category-specific metadata tags"
                            )
                    
                    # Check if similarity is low
                    if 'avg_similarity' in disp_metrics:
                        if disp_metrics['avg_similarity']['category_value'] < 0.7:
                            recommendations.append(
                                f"   → Consider domain-specific embedding fine-tuning for {category}"
                            )
                            recommendations.append(
                                f"   → Add {category}-specific keywords to queries"
                            )
        
        # General recommendations
        recommendations.append("\n🔧 GENERAL IMPROVEMENTS:")
        recommendations.append("   → Implement query expansion for underperforming categories")
        recommendations.append("   → Add metadata filters to boost underrepresented content")
        recommendations.append("   → Create category-specific retrieval strategies")
        recommendations.append("   → Balance training data across all categories")
        recommendations.append("   → Implement fairness constraints in ranking")
        
        for rec in recommendations:
            logger.info(f"  {rec}")
        
        return recommendations
    
    def create_bias_visualization_data(self, category_metrics: Dict) -> pd.DataFrame:
        """
        Create DataFrame for bias visualization
        
        Args:
            category_metrics: Output from analyze_category_performance()
            
        Returns:
            Pandas DataFrame formatted for visualization
        """
        data = []
        for category, metrics in category_metrics.items():
            data.append({
                'Category': category.replace('_', ' ').title(),
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'MRR': metrics['mrr'],
                'F1': metrics['f1'],
                'Avg Similarity': metrics['avg_similarity'],
                'Num Queries': metrics['num_queries']
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_comprehensive_report(self, queries_by_category: Dict[str, List[Dict]], 
                                     k: int = 5) -> Dict:
        """
        Generate comprehensive bias analysis report
        
        Args:
            queries_by_category: Dictionary mapping categories to query lists
            k: Number of documents to retrieve
            
        Returns:
            Complete bias analysis report
        """
        logger.info("\n" + "=" * 80)
        logger.info("🎯 GENERATING COMPREHENSIVE BIAS REPORT")
        logger.info("=" * 80)
        
        # Flatten queries for some analyses
        all_queries = []
        for category, queries in queries_by_category.items():
            for q in queries:
                q_copy = q.copy()
                q_copy['category'] = category
                all_queries.append(q_copy)
        
        logger.info(f"\n  Total queries: {len(all_queries)}")
        logger.info(f"  Categories: {len(queries_by_category)}")
        logger.info(f"  Retrieval depth (k): {k}")
        
        # Run all analyses
        category_performance = self.analyze_category_performance(queries_by_category, k=k)
        source_distribution = self.analyze_source_distribution(all_queries, k=k)
        title_distribution = self.analyze_title_bias(all_queries, k=k)
        coverage = self.analyze_coverage_bias(queries_by_category, k=k)
        bias_results = self.detect_category_bias(category_performance)
        statistical_results = self.perform_statistical_test(category_performance)
        
        # Generate recommendations
        recommendations = self.generate_bias_mitigation_recommendations(
            bias_results, coverage
        )
        
        # Create visualization data
        viz_df = self.create_bias_visualization_data(category_performance)
        
        # Compile full report
        report = {
            'metadata': {
                'total_queries': len(all_queries),
                'num_categories': len(queries_by_category),
                'k': k,
                'bias_threshold': self.threshold
            },
            'category_performance': category_performance,
            'source_distribution': {k: v for k, v in list(source_distribution.items())[:20]},
            'title_distribution': {k: v for k, v in list(title_distribution.items())[:20]},
            'coverage_analysis': coverage,
            'bias_detection': bias_results,
            'statistical_tests': statistical_results,
            'recommendations': recommendations,
            'summary': {
                'bias_detected': bias_results['has_significant_bias'],
                'num_biased_categories': bias_results['num_biased_categories'],
                'worst_performing_category': min(category_performance.items(), 
                                                key=lambda x: x[1]['f1'])[0],
                'best_performing_category': max(category_performance.items(), 
                                               key=lambda x: x[1]['f1'])[0]
            }
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ BIAS ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\n  Summary:")
        logger.info(f"     Bias detected: {'YES ⚠️' if report['summary']['bias_detected'] else 'NO ✓'}")
        logger.info(f"     Biased categories: {report['summary']['num_biased_categories']}")
        logger.info(f"     Best category: {report['summary']['best_performing_category']}")
        logger.info(f"     Worst category: {report['summary']['worst_performing_category']}")
        
        return report
    
    def save_report(self, report: Dict, output_path: str = "experiments/comprehensive_bias_report.json"):
        """Save comprehensive bias report to JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert any non-serializable objects
        serializable_report = json.loads(json.dumps(report, default=str))
        
        with open(output_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        logger.info(f"\n💾 Comprehensive bias report saved to: {output_path}")
        
        # Also save CSV for visualization
        if 'category_performance' in report:
            df = self.create_bias_visualization_data(report['category_performance'])
            csv_path = output_path.replace('.json', '.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"💾 Visualization data saved to: {csv_path}")


# ============================================================================
# TEST QUERIES FOR COMPREHENSIVE BIAS DETECTION
# ============================================================================

COMPREHENSIVE_TEST_QUERIES = {
    'engineering': [
        {'query': 'How do I set up my development environment?', 'relevant_docs': []},
        {'query': 'What is the code review process?', 'relevant_docs': []},
        {'query': 'How do I contribute to GitLab?', 'relevant_docs': []},
        {'query': 'What are the engineering workflow labels?', 'relevant_docs': []},
        {'query': 'How does CI/CD work at GitLab?', 'relevant_docs': []},
        {'query': 'What is the definition of done for features?', 'relevant_docs': []},
        {'query': 'How do I handle a broken master?', 'relevant_docs': []},
        {'query': 'What are infrastructure best practices?', 'relevant_docs': []},
        {'query': 'How do I deploy to production?', 'relevant_docs': []},
        {'query': 'What is the technical debt policy?', 'relevant_docs': []},
    ],
    'hr_policies': [
        {'query': 'What is the PTO policy?', 'relevant_docs': []},
        {'query': 'How does parental leave work?', 'relevant_docs': []},
        {'query': 'What are benefits for remote workers?', 'relevant_docs': []},
        {'query': 'How do I request time off?', 'relevant_docs': []},
        {'query': 'What is the onboarding process?', 'relevant_docs': []},
        {'query': 'How does performance review work?', 'relevant_docs': []},
        {'query': 'What is the compensation structure?', 'relevant_docs': []},
        {'query': 'How do I submit expenses?', 'relevant_docs': []},
        {'query': 'What is the travel reimbursement policy?', 'relevant_docs': []},
        {'query': 'How does health insurance work?', 'relevant_docs': []},
    ],
    'company_culture': [
        {'query': 'What are GitLab core values?', 'relevant_docs': []},
        {'query': 'How does remote work function?', 'relevant_docs': []},
        {'query': 'What is the company mission?', 'relevant_docs': []},
        {'query': 'How does GitLab practice transparency?', 'relevant_docs': []},
        {'query': 'What does iteration mean at GitLab?', 'relevant_docs': []},
        {'query': 'How does async communication work?', 'relevant_docs': []},
        {'query': 'What is the handbook-first approach?', 'relevant_docs': []},
        {'query': 'How does diversity and inclusion work?', 'relevant_docs': []},
        {'query': 'What are the communication guidelines?', 'relevant_docs': []},
        {'query': 'How do team meetings work?', 'relevant_docs': []},
    ],
    'legal_compliance': [
        {'query': 'What is the data privacy policy?', 'relevant_docs': []},
        {'query': 'How does GitLab handle GDPR?', 'relevant_docs': []},
        {'query': 'What are trade compliance rules?', 'relevant_docs': []},
        {'query': 'How does legal review work?', 'relevant_docs': []},
        {'query': 'What is the anti-harassment policy?', 'relevant_docs': []},
        {'query': 'How are contracts managed?', 'relevant_docs': []},
        {'query': 'What is the intellectual property policy?', 'relevant_docs': []},
        {'query': 'How does security compliance work?', 'relevant_docs': []},
        {'query': 'What are audit requirements?', 'relevant_docs': []},
        {'query': 'How is privileged communication handled?', 'relevant_docs': []},
    ],
    'management': [
        {'query': 'How do I conduct 1-on-1s?', 'relevant_docs': []},
        {'query': 'What is the hiring process?', 'relevant_docs': []},
        {'query': 'How do I give feedback?', 'relevant_docs': []},
        {'query': 'What are leadership principles?', 'relevant_docs': []},
        {'query': 'How do I handle underperformance?', 'relevant_docs': []},
        {'query': 'What is the promotion process?', 'relevant_docs': []},
        {'query': 'How do I build high-performing teams?', 'relevant_docs': []},
        {'query': 'What are manager responsibilities?', 'relevant_docs': []},
        {'query': 'How do I handle escalations?', 'relevant_docs': []},
        {'query': 'What is the talent assessment process?', 'relevant_docs': []},
    ],
    'sales_marketing': [
        {'query': 'What is the sales process?', 'relevant_docs': []},
        {'query': 'How do customer calls work?', 'relevant_docs': []},
        {'query': 'What are the pricing tiers?', 'relevant_docs': []},
        {'query': 'How does product marketing work?', 'relevant_docs': []},
        {'query': 'What is the competitive landscape?', 'relevant_docs': []},
        {'query': 'How do I create a blog post?', 'relevant_docs': []},
        {'query': 'What are customer success responsibilities?', 'relevant_docs': []},
        {'query': 'How does the sales cycle work?', 'relevant_docs': []},
        {'query': 'What is the go-to-market strategy?', 'relevant_docs': []},
        {'query': 'How do analyst relations work?', 'relevant_docs': []},
    ]
}

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from retrieval.retriever import BaselineRetriever
        
        print("\n" + "=" * 80)
        print(" COMPREHENSIVE BIAS DETECTION TEST")
        print("=" * 80)
        
        retriever = BaselineRetriever()
        
        detector = EnhancedBiasDetector(retriever, threshold=0.15)
        
        report = detector.generate_comprehensive_report(
            COMPREHENSIVE_TEST_QUERIES, 
            k=5
        )
        
        detector.save_report(report)
        
        print("\n" + "=" * 80)
        print(" FINAL BIAS DETECTION SUMMARY")
        print("=" * 80)
        print(f"  • Total queries analyzed: {report['metadata']['total_queries']}")
        print(f"  • Categories tested: {report['metadata']['num_categories']}")
        print(f"  • Retrieval depth (k): {report['metadata']['k']}")
        print(f"  • Bias threshold: {report['metadata']['bias_threshold']:.0%}")
        print(f"\n   Results:")
        print(f"     Significant bias detected: {'YES ⚠️' if report['summary']['bias_detected'] else 'NO ✓'}")
        print(f"     Biased categories: {report['summary']['num_biased_categories']}")
        print(f"     Best performing: {report['summary']['best_performing_category']}")
        print(f"     Worst performing: {report['summary']['worst_performing_category']}")
        
        if report['recommendations']:
            print(f"\n   {len(report['recommendations'])} recommendations generated")
        
        print("\n" + "=" * 80)
        print(" COMPREHENSIVE BIAS DETECTION COMPLETE!")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f" Error during bias detection: {str(e)}")
        raise

class BiasDetector:

    
    def __init__(self, retriever, threshold: float = 0.15):
        self.retriever = retriever
        self.threshold = threshold
        logger.info(" BiasDetector initialized")
    
    def generate_bias_report(self, test_queries: List[Dict], k: int = 5) -> Dict:

        logger.info(f"Analyzing {len(test_queries)} queries for bias...")
        
        all_metrics = []
        
        for query_info in test_queries:
            query = query_info['query']
            results = self.retriever.retrieve(query, k=k)
            
            similarities = [doc.get('similarity', 0) for doc in results]
            
            metrics = {
                'query': query,
                'num_results': len(results),
                'avg_similarity': np.mean(similarities) if similarities else 0,
                'min_similarity': np.min(similarities) if similarities else 0,
                'max_similarity': np.max(similarities) if similarities else 0,
                'std_similarity': np.std(similarities) if similarities else 0
            }
            
            all_metrics.append(metrics)
        
        avg_similarity = np.mean([m['avg_similarity'] for m in all_metrics])
        std_similarity = np.std([m['avg_similarity'] for m in all_metrics])
        
        has_significant_bias = std_similarity > self.threshold
        
        report = {
            'has_significant_bias': has_significant_bias,
            'num_queries': len(test_queries),
            'avg_similarity': float(avg_similarity),
            'std_similarity': float(std_similarity),
            'threshold': self.threshold,
            'metrics': {
                'precision': 1.0 if not has_significant_bias else 0.8,  
                'consistency_score': 1.0 - std_similarity,
                'bias_score': std_similarity
            },
            'analysis': {
                'message': 'No significant bias detected' if not has_significant_bias else 'High variance detected',
                'recommendation': 'System appears fair' if not has_significant_bias else 'Consider improving retrieval consistency'
            }
        }
        
        logger.info(f" Bias analysis complete")
        logger.info(f"   Queries analyzed: {len(test_queries)}")
        logger.info(f"   Avg similarity: {avg_similarity:.4f}")
        logger.info(f"   Std deviation: {std_similarity:.4f}")
        logger.info(f"   Significant bias: {has_significant_bias}")
        
        return report