from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity
)
from datasets import Dataset
import pandas as pd
import numpy as np
from typing import List, Dict
import logging
import json
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGASEvaluator:
    
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        
        self.api_key = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.llm = None
        
        if self.api_key:
            try:
                from groq import Groq
                groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                self.# Using Groq instead of gemini-2.0-flash
# llm = groq_client
                logger.info("RAGAS Evaluator initialized with Gemini API (gemini-2.0-flash)")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                logger.info("RAGAS Evaluator initialized without LLM")
        else:
            logger.info("RAGAS Evaluator initialized without API key")
        
        logger.info(f"Metrics: {[m.name for m in self.metrics]}")
    
    def evaluate_retrieval_only(self, 
                                retriever,
                                test_data: List[Dict]) -> Dict:
        
        logger.info("\n" + "="*80)
        logger.info("RAGAS EVALUATION (RETRIEVAL-ONLY MODE)")
        logger.info("="*80)
        
        results_data = []
        
        for item in test_data:
            question = item['question']
            ground_truth = item.get('ground_truth', '')
            
            logger.info(f"\nEvaluating: {question[:60]}...")
            
            retrieved_docs = retriever.retrieve(question, k=5)
            contexts = [doc['document'] for doc in retrieved_docs]
            
            if self.llm and contexts:
                try:
                    prompt = f"Based on these documents, answer the question: {question}\n\nDocuments:\n{contexts[0][:500]}"
                    response = self.groq_client.chat.completions.create(model="mixtral-8x7b-32768", messages=[{"role": "user", "content": prompt}])
                    synthetic_answer = response.choices[0].message.content
                    logger.info("Generated answer using Gemini")
                except Exception as e:
                    logger.warning(f"Gemini generation failed: {e}")
                    synthetic_answer = f"Based on the retrieved documents: {contexts[0][:200]}..."
            else:
                synthetic_answer = f"Based on the retrieved documents: {contexts[0][:200]}..."
            
            results_data.append({
                'question': question,
                'answer': synthetic_answer,
                'contexts': contexts,
                'ground_truth': ground_truth
            })
        
        dataset = Dataset.from_pandas(pd.DataFrame(results_data))
        
        logger.info("\nRunning RAGAS evaluation...")
        
        context_metrics = [context_precision, context_recall]
        
        try:
            if self.api_key:
                evaluation_results = evaluate(
                    dataset,
                    metrics=context_metrics
                )
                
                logger.info("RAGAS evaluation complete!")
                
                results_dict = {
                    'num_queries': len(test_data),
                    'metrics': {},
                    'using_gemini': True
                }
                
                for metric_name, value in evaluation_results.items():
                    results_dict['metrics'][metric_name] = float(value)
                    logger.info(f"{metric_name}: {value:.4f}")
                
                return results_dict
            else:
                raise Exception("No API key available")
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if 'api_key' in error_msg or 'api' in error_msg:
                logger.info("RAGAS requires LLM API for full evaluation")
                logger.info("Using custom context-based metrics instead")
            else:
                logger.warning(f"RAGAS evaluation error: {e}")
                logger.info("Falling back to custom metrics...")
            
            return self._custom_context_metrics(results_data)
    
    def _custom_context_metrics(self, results_data: List[Dict]) -> Dict:
        
        logger.info("Computing custom context metrics...")
        
        metrics = {
            'num_queries': len(results_data),
            'avg_contexts_retrieved': float(np.mean([len(item['contexts']) for item in results_data])),
            'context_coverage': float(self._calculate_context_coverage(results_data)),
            'context_diversity': float(self._calculate_context_diversity(results_data))
        }
        
        logger.info("Custom metrics computed:")
        logger.info(f"Avg contexts: {metrics['avg_contexts_retrieved']:.2f}")
        logger.info(f"Context coverage: {metrics['context_coverage']:.4f}")
        logger.info(f"Context diversity: {metrics['context_diversity']:.4f}")
        
        return {
            'metrics': metrics, 
            'num_queries': len(results_data),
            'using_gemini': False
        }
    
    def _calculate_context_coverage(self, results_data: List[Dict]) -> float:
        
        coverages = []
        for item in results_data:
            question_tokens = set(item['question'].lower().split())
            context_tokens = set()
            
            for context in item['contexts']:
                context_tokens.update(context.lower().split())
            
            coverage = len(question_tokens & context_tokens) / len(question_tokens) if question_tokens else 0
            coverages.append(coverage)
        
        return np.mean(coverages) if coverages else 0.0
    
    def _calculate_context_diversity(self, results_data: List[Dict]) -> float:
        
        diversities = []
        for item in results_data:
            contexts = item['contexts']
            if len(contexts) < 2:
                diversities.append(1.0)
                continue
            
            all_words = []
            for context in contexts:
                all_words.extend(context.lower().split())
            
            unique_ratio = len(set(all_words)) / len(all_words) if all_words else 0
            diversities.append(unique_ratio)
        
        return np.mean(diversities) if diversities else 0.0
    
    def save_results(self, results: Dict, output_path: str = "experiments/ragas_evaluation.json"):
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nRAGAS results saved: {output_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from retrieval.advanced_retriever import AdvancedRetriever
    
    print("\n" + "="*80)
    print("TESTING RAGAS EVALUATION FRAMEWORK")
    print("="*80)
    
    retriever = AdvancedRetriever()
    evaluator = RAGASEvaluator()
    
    test_data = [
        {
            'question': 'What is GitLab\'s approach to sustainability?',
            'ground_truth': 'GitLab focuses on corporate sustainability through environmental and social responsibility.'
        },
        {
            'question': 'How does risk management work at GitLab?',
            'ground_truth': 'GitLab has a Risk Management and Dispute Resolution division that handles risk strategies.'
        },
        {
            'question': 'What are the legal compliance requirements?',
            'ground_truth': 'GitLab maintains legal compliance through their Legal and Corporate Affairs team.'
        }
    ]
    
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        print("\nGemini API key found - using gemini-2.0-flash for evaluation")
    else:
        print("\nNo API key - using custom context-based metrics")
    
    results = evaluator.evaluate_retrieval_only(retriever, test_data)
    
    print("\n" + "="*80)
    print("RAGAS EVALUATION RESULTS")
    print("="*80)
    
    if 'metrics' in results:
        for metric_name, value in results['metrics'].items():
            if isinstance(value, (int, float)):
                print(f"{metric_name}: {value:.4f}")
            else:
                print(f"{metric_name}: {value}")
    
    print(f"\nTotal queries evaluated: {results.get('num_queries', 0)}")
    print(f"Using Gemini: {results.get('using_gemini', False)}")
    
    evaluator.save_results(results)
    
    print("\n" + "="*80)
    print("RAGAS EVALUATION COMPLETE")
    print("="*80 + "\n")