import mlflow
import mlflow.pyfunc
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentTracker:
    
    def __init__(self, 
                 experiment_name: str = "onboarding-assistant-retrieval",
                 tracking_uri: str = "file:./experiments/mlruns"):

        mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        logger.info(f" MLflow initialized")
        logger.info(f"   Experiment: {experiment_name}")
        logger.info(f"   Tracking URI: {tracking_uri}")
    
    def log_retrieval_experiment(self,
                                 model_name: str,
                                 embedding_model: str,
                                 vector_store: str,
                                 eval_results: Dict,
                                 params: Optional[Dict] = None,
                                 tags: Optional[Dict] = None):

        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"\n{'='*80}")
            logger.info(f" LOGGING EXPERIMENT: {run_name}")
            logger.info(f"{'='*80}")
            
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("embedding_model", embedding_model)
            mlflow.log_param("vector_store", vector_store)
            mlflow.log_param("num_test_queries", eval_results['num_queries'])
            
            if params:
                for key, value in params.items():
                    mlflow.log_param(key, value)
            
            mlflow.log_metric("mrr", eval_results['mrr'])
            
            for k_label, metrics in eval_results['metrics_by_k'].items():
                k_value = k_label.split('=')[1]
                mlflow.log_metric(f"precision_at_{k_value}", metrics['precision'])
                mlflow.log_metric(f"recall_at_{k_value}", metrics['recall'])
                mlflow.log_metric(f"f1_at_{k_value}", metrics['f1'])
                mlflow.log_metric(f"ndcg_at_{k_value}", metrics['ndcg'])
            
            if tags:
                mlflow.set_tags(tags)
            
            mlflow.set_tag("timestamp", datetime.now().isoformat())
            
            results_file = Path("experiments/temp_results.json")
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            mlflow.log_artifact(str(results_file), artifact_path="evaluation")
            results_file.unlink()  
            
            logger.info(f"\n Experiment logged successfully!")
            logger.info(f"   Run ID: {run.info.run_id}")
            logger.info(f"   Run Name: {run_name}")
            logger.info(f"\n{'='*80}\n")
            
            return run.info.run_id
    
    def log_embedding_generation(self,
                                model_name: str,
                                num_documents: int,
                                embedding_dim: int,
                                processing_time: float):

        run_name = f"embedding_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("embedding_model", model_name)
            mlflow.log_param("num_documents", num_documents)
            mlflow.log_param("embedding_dimension", embedding_dim)
            mlflow.log_metric("processing_time_seconds", processing_time)
            mlflow.log_metric("docs_per_second", num_documents / processing_time if processing_time > 0 else 0)
            
            mlflow.set_tag("experiment_type", "embedding_generation")
            mlflow.set_tag("timestamp", datetime.now().isoformat())
            
            logger.info(f" Embedding generation logged: {run_name}")
    
    def compare_experiments(self, run_ids: list):
 
        logger.info(f"\n{'='*80}")
        logger.info(f" COMPARING {len(run_ids)} EXPERIMENTS")
        logger.info(f"{'='*80}\n")
        
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            
            print(f"\n Run: {run.data.tags.get('mlflow.runName', run_id)}")
            print(f"   Run ID: {run_id}")
            print(f"   MRR: {run.data.metrics.get('mrr', 'N/A'):.4f}")
            print(f"   Precision@5: {run.data.metrics.get('precision_at_5', 'N/A'):.4f}")
            print(f"   Recall@5: {run.data.metrics.get('recall_at_5', 'N/A'):.4f}")
            print(f"   Embedding Model: {run.data.params.get('embedding_model', 'N/A')}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from retrieval.retriever import BaselineRetriever
    from evaluation.metrics import RetrievalEvaluator
    
    print("\n" + "=" * 80)
    print(" TESTING MLFLOW EXPERIMENT TRACKING")
    print("=" * 80)
    
    retriever = BaselineRetriever()
    evaluator = RetrievalEvaluator(retriever)
    tracker = ExperimentTracker()
    
    test_queries = [
        {
            'query': 'What is GitLab\'s approach to sustainability?',
            'relevant_ids': ['doc_8']
        },
        {
            'query': 'How does risk management work at GitLab?',
            'relevant_ids': ['doc_12']
        },
        {
            'query': 'Tell me about CI/CD processes',
            'relevant_ids': ['doc_3', 'doc_15']
        }
    ]
    
    print("\n Running evaluation...")
    results = evaluator.evaluate(test_queries, k_values=[1, 3, 5])
    
    print("\n Logging to MLflow...")
    run_id = tracker.log_retrieval_experiment(
        model_name="baseline-retriever-v1",
        embedding_model="all-mpnet-base-v2",
        vector_store="chromadb",
        eval_results=results,
        params={
            "chunk_size": 19,
            "embedding_dim": 384
        },
        tags={
            "experiment_type": "baseline_retrieval",
            "status": "test"
        }
    )
    
    print("\n" + "=" * 80)
    print(" MLFLOW TRACKING TEST COMPLETE!")
    print("=" * 80)
    print(f"\n Run ID: {run_id}")
    print(f"\n To view results in MLflow UI, run:")
    print(f"   mlflow ui --backend-store-uri file:./experiments/mlruns")
    print(f"   Then open: http://localhost:5000")
    print("\n" + "=" * 80)