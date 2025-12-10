"""
Metrics Collector for Model Performance Monitoring
Collects and stores key performance metrics for RAG pipeline
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict
import pickle
from google.cloud import storage
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Data class for storing model performance metrics"""
    timestamp: str
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    f1_at_5: float
    f1_at_10: float
    mrr: float
    ndcg_at_10: float
    avg_response_time: float
    query_count: int
    avg_relevance_score: float
    
    def to_dict(self):
        return asdict(self)


class MetricsCollector:
    """Collects and manages model performance metrics"""
    
    def __init__(self, 
                 metrics_dir: str = "experiments/monitoring",
                 gcs_bucket: Optional[str] = None,
                 use_gcs: bool = False):
        """
        Initialize MetricsCollector
        
        Args:
            metrics_dir: Directory to store metrics
            gcs_bucket: GCS bucket name for remote storage
            use_gcs: Whether to use GCS for storage
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.gcs_bucket = gcs_bucket
        self.use_gcs = use_gcs
        self.gcs_client = None
        
        if self.use_gcs and self.gcs_bucket:
            self.gcs_client = storage.Client()
        
        self.metrics_file = self.metrics_dir / "metrics_history.json"
        self.metrics_raw_file = self.metrics_dir / "metrics_raw.pkl"
        
        # Initialize metrics history
        self.metrics_history = self._load_metrics_history()
    
    def _load_metrics_history(self) -> List[Dict[str, Any]]:
        """Load existing metrics history"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics history: {e}")
                return []
        return []
    
    def collect_query_metrics(self,
                            query: str,
                            retrieved_ids: List[str],
                            relevant_ids: List[str],
                            response_time: float,
                            relevance_scores: List[float],
                            generation_time: float = 0.0) -> Dict[str, Any]:
        """
        Collect metrics for a single query
        
        Args:
            query: The user query
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs (ground truth)
            response_time: Time taken for response (in seconds)
            relevance_scores: Relevance scores for retrieved documents
            generation_time: Time taken for LLM generation
            
        Returns:
            Dictionary of computed metrics
        """
        from src.evaluation.metrics import RetrievalMetrics
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response_time": response_time,
            "generation_time": generation_time,
            "precision_at_5": RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, 5),
            "precision_at_10": RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, 10),
            "recall_at_5": RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, 5),
            "recall_at_10": RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, 10),
            "f1_at_5": RetrievalMetrics.f1_at_k(retrieved_ids, relevant_ids, 5),
            "f1_at_10": RetrievalMetrics.f1_at_k(retrieved_ids, relevant_ids, 10),
            "mrr": RetrievalMetrics.mean_reciprocal_rank([retrieved_ids], [relevant_ids]),
            "ndcg_at_10": RetrievalMetrics.ndcg_at_k(retrieved_ids, relevant_ids, 10),
            "avg_relevance_score": float(np.mean(relevance_scores)) if relevance_scores else 0.0,
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(relevant_ids)
        }
        
        return metrics
    
    def aggregate_metrics(self, 
                         metrics_list: List[Dict[str, Any]],
                         time_window: str = "daily") -> ModelMetrics:
        """
        Aggregate individual query metrics into overall performance metrics
        
        Args:
            metrics_list: List of individual query metrics
            time_window: Time window for aggregation ('hourly', 'daily', 'weekly')
            
        Returns:
            ModelMetrics object with aggregated metrics
        """
        if not metrics_list:
            raise ValueError("Metrics list cannot be empty")
        
        # Extract numeric metrics
        precision_5 = [m.get("precision_at_5", 0) for m in metrics_list]
        precision_10 = [m.get("precision_at_10", 0) for m in metrics_list]
        recall_5 = [m.get("recall_at_5", 0) for m in metrics_list]
        recall_10 = [m.get("recall_at_10", 0) for m in metrics_list]
        f1_5 = [m.get("f1_at_5", 0) for m in metrics_list]
        f1_10 = [m.get("f1_at_10", 0) for m in metrics_list]
        mrr = [m.get("mrr", 0) for m in metrics_list]
        ndcg = [m.get("ndcg_at_10", 0) for m in metrics_list]
        response_times = [m.get("response_time", 0) for m in metrics_list]
        relevance_scores = [m.get("avg_relevance_score", 0) for m in metrics_list]
        
        aggregated = ModelMetrics(
            timestamp=datetime.now().isoformat(),
            precision_at_5=float(np.mean(precision_5)),
            precision_at_10=float(np.mean(precision_10)),
            recall_at_5=float(np.mean(recall_5)),
            recall_at_10=float(np.mean(recall_10)),
            f1_at_5=float(np.mean(f1_5)),
            f1_at_10=float(np.mean(f1_10)),
            mrr=float(np.mean(mrr)),
            ndcg_at_10=float(np.mean(ndcg)),
            avg_response_time=float(np.mean(response_times)),
            query_count=len(metrics_list),
            avg_relevance_score=float(np.mean(relevance_scores))
        )
        
        return aggregated
    
    def save_metrics(self, 
                    metrics,
                    save_raw: bool = True) -> None:
        """
        Save metrics to file (JSON and pickle)
        
        Args:
            metrics: Metrics dictionary or ModelMetrics object
            save_raw: Whether to save raw pickle format
        """
        # Convert to dict if ModelMetrics
        if isinstance(metrics, ModelMetrics):
            metrics_dict = metrics.to_dict()
        else:
            metrics_dict = metrics
        
        # Add to history
        if isinstance(metrics_dict, list):
            self.metrics_history.extend(metrics_dict)
        else:
            self.metrics_history.append(metrics_dict)
        
        # Save to JSON
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
            logger.info(f"Metrics saved to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
        
        # Save raw pickle format
        if save_raw:
            try:
                with open(self.metrics_raw_file, 'wb') as f:
                    pickle.dump(self.metrics_history, f)
            except Exception as e:
                logger.error(f"Error saving raw metrics: {e}")
        
        # Upload to GCS if enabled
        if self.use_gcs and self.gcs_client:
            self._upload_to_gcs(metrics_dict)
    
    def _upload_to_gcs(self, metrics: Dict[str, Any]) -> None:
        """Upload metrics to GCS"""
        try:
            bucket = self.gcs_client.bucket(self.gcs_bucket)
            blob = bucket.blob(f"monitoring/metrics/{datetime.now().isoformat()}.json")
            blob.upload_from_string(
                json.dumps(metrics, default=str),
                content_type='application/json'
            )
            logger.info(f"Metrics uploaded to GCS: {self.gcs_bucket}")
        except Exception as e:
            logger.error(f"Error uploading to GCS: {e}")
    
    def get_metrics_history(self, 
                           limit: Optional[int] = None,
                           start_time: Optional[str] = None,
                           end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get metrics history with optional filtering
        
        Args:
            limit: Maximum number of records to return
            start_time: ISO format start timestamp
            end_time: ISO format end timestamp
            
        Returns:
            List of metrics dictionaries
        """
        history = self.metrics_history.copy()
        
        # Filter by time range
        if start_time or end_time:
            filtered = []
            for metric in history:
                ts = metric.get("timestamp", "")
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                filtered.append(metric)
            history = filtered
        
        # Limit results
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the latest metrics entry"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def compute_metric_statistics(self,
                                 metric_name: str,
                                 limit: Optional[int] = None) -> Dict[str, float]:
        """
        Compute statistics for a specific metric
        
        Args:
            metric_name: Name of the metric (e.g., 'precision_at_5')
            limit: Limit to recent N records
            
        Returns:
            Dictionary with mean, std, min, max
        """
        history = self.get_metrics_history(limit=limit)
        values = [m.get(metric_name, 0) for m in history if metric_name in m]
        
        if not values:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0
            }
        
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values)
        }
    
    def export_metrics(self, export_path: str, format: str = "json") -> None:
        """
        Export metrics to file in specified format
        
        Args:
            export_path: Path to export file
            format: Export format ('json' or 'csv')
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "json":
                with open(export_path, 'w') as f:
                    json.dump(self.metrics_history, f, indent=2, default=str)
            elif format == "csv":
                import pandas as pd
                df = pd.DataFrame(self.metrics_history)
                df.to_csv(export_path, index=False)
            logger.info(f"Metrics exported to {export_path}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
