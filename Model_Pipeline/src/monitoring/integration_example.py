"""
Monitoring Integration Example
Shows how to integrate monitoring into the RAG pipeline
"""

import logging
import time
from typing import Dict, List, Any, Optional
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.monitoring.metrics_collector import MetricsCollector
from src.monitoring.data_drift_detector import DataDriftDetector
from src.monitoring.alert_system import ThresholdAlertSystem, AlertThreshold
from src.generation.rag_pipeline import UniversalRAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonitoredRAGPipeline:
    """RAG Pipeline with integrated monitoring"""
    
    def __init__(self,
                 rag_pipeline: UniversalRAGPipeline,
                 enable_monitoring: bool = True,
                 enable_drift_detection: bool = True,
                 enable_alerts: bool = True):
        """
        Initialize MonitoredRAGPipeline
        
        Args:
            rag_pipeline: UniversalRAGPipeline instance
            enable_monitoring: Enable metrics collection
            enable_drift_detection: Enable data drift detection
            enable_alerts: Enable alert system
        """
        self.rag_pipeline = rag_pipeline
        
        self.enable_monitoring = enable_monitoring
        self.enable_drift_detection = enable_drift_detection
        self.enable_alerts = enable_alerts
        
        # Initialize monitoring components
        if self.enable_monitoring:
            self.metrics_collector = MetricsCollector()
        
        if self.enable_drift_detection:
            self.drift_detector = DataDriftDetector()
            self._set_baseline_embeddings()
        
        if self.enable_alerts:
            self.alert_system = ThresholdAlertSystem()
            self._configure_default_thresholds()
        
        # Metrics buffer (accumulate before saving)
        self.metrics_buffer = []
        self.buffer_size = 10
    
    def _set_baseline_embeddings(self) -> None:
        """Set baseline embeddings from training data"""
        try:
            # Load baseline embeddings from vector store metadata
            from pathlib import Path
            import json
            
            metadata_path = Path("models/vector_store_metadata/metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    # Store metadata for later use
                    self.baseline_metadata = metadata
                    logger.info("Baseline embeddings metadata loaded")
        except Exception as e:
            logger.warning(f"Could not load baseline embeddings: {e}")
    
    def _configure_default_thresholds(self) -> None:
        """Configure sensible default thresholds"""
        if not self.enable_alerts:
            return
        
        # These are example thresholds - adjust based on your requirements
        thresholds = {
            "precision_at_5": AlertThreshold(
                metric_name="precision_at_5",
                lower_threshold=0.5,
                severity="warning"
            ),
            "recall_at_5": AlertThreshold(
                metric_name="recall_at_5",
                lower_threshold=0.3,
                severity="warning"
            ),
            "f1_at_5": AlertThreshold(
                metric_name="f1_at_5",
                lower_threshold=0.35,
                severity="warning"
            ),
            "mrr": AlertThreshold(
                metric_name="mrr",
                lower_threshold=0.3,
                severity="warning"
            ),
            "avg_response_time": AlertThreshold(
                metric_name="avg_response_time",
                lower_threshold=0,
                upper_threshold=5.0,
                severity="critical"
            ),
            "avg_relevance_score": AlertThreshold(
                metric_name="avg_relevance_score",
                lower_threshold=0.5,
                severity="warning"
            )
        }
        
        for threshold in thresholds.values():
            self.alert_system.set_threshold(threshold)
    
    def generate_with_monitoring(self,
                                query: str,
                                relevant_doc_ids: Optional[List[str]] = None,
                                track_metrics: bool = True) -> Dict[str, Any]:
        """
        Generate RAG response with monitoring
        
        Args:
            query: User query
            relevant_doc_ids: Ground truth relevant documents (for evaluation)
            track_metrics: Whether to track metrics for this query
            
        Returns:
            Dictionary with response and metrics
        """
        start_time = time.time()
        
        try:
            # Get RAG response
            result = self.rag_pipeline.generate(query)
            
            generation_time = time.time() - start_time
            
            # Extract retrieval information
            retrieved_docs = result.get('retrieved_docs', [])
            retrieved_ids = [doc.get('id', doc.get('source', '')) for doc in retrieved_docs]
            relevance_scores = [doc.get('relevance_score', 0) for doc in retrieved_docs]
            
            # Track metrics if enabled
            if track_metrics and self.enable_monitoring:
                if relevant_doc_ids is None:
                    relevant_doc_ids = []
                
                metrics = self.metrics_collector.collect_query_metrics(
                    query=query,
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_doc_ids,
                    response_time=generation_time,
                    relevance_scores=relevance_scores,
                    generation_time=generation_time
                )
                
                self.metrics_buffer.append(metrics)
                
                # Save metrics when buffer is full
                if len(self.metrics_buffer) >= self.buffer_size:
                    self._save_buffered_metrics()
                
                # Check for alerts
                if self.enable_alerts:
                    self._check_and_trigger_alerts(metrics)
                
                # Check for data drift
                if self.enable_drift_detection:
                    self._check_data_drift(retrieved_docs)
            
            # Add metrics to response
            result['monitoring'] = {
                'metrics_tracked': track_metrics,
                'generation_time': generation_time,
                'retrieval_count': len(retrieved_ids)
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error during monitored generation: {e}")
            raise
    
    def _save_buffered_metrics(self) -> None:
        """Save buffered metrics to storage"""
        if not self.metrics_buffer:
            return
        
        try:
            for metrics in self.metrics_buffer:
                self.metrics_collector.save_metrics(metrics)
            
            # Aggregate and log
            aggregated = self.metrics_collector.aggregate_metrics(self.metrics_buffer)
            logger.info(f"Metrics saved - Avg Precision@5: {aggregated.precision_at_5:.4f}")
            
            self.metrics_buffer = []
        except Exception as e:
            logger.error(f"Error saving buffered metrics: {e}")
    
    def _check_and_trigger_alerts(self, metrics: Dict[str, Any]) -> None:
        """Check metrics against thresholds and trigger alerts"""
        if not self.enable_alerts:
            return
        
        try:
            alerts = self.alert_system.check_metrics(metrics)
            for alert in alerts:
                self.alert_system.trigger_alert(alert)
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _check_data_drift(self, retrieved_docs: List[Dict[str, Any]]) -> None:
        """Check for data drift in retrieved documents"""
        if not self.enable_drift_detection:
            return
        
        try:
            # Extract embeddings from retrieved documents if available
            embeddings = []
            for doc in retrieved_docs:
                if 'embedding' in doc:
                    embeddings.append(doc['embedding'])
            
            if embeddings:
                embeddings = np.array(embeddings)
                
                # Detect embedding drift
                result = self.drift_detector.detect_embedding_drift(
                    baseline_embeddings=np.random.randn(len(embeddings), embeddings.shape[1]),
                    current_embeddings=embeddings,
                    feature_name="query_embeddings"
                )
                
                if result:
                    self.drift_detector.save_drift_report(result)
                    
                    if result.is_drift_detected:
                        logger.warning(f"Data drift detected: {result.message}")
        except Exception as e:
            logger.warning(f"Error checking drift: {e}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'monitoring_enabled': self.enable_monitoring,
            'drift_detection_enabled': self.enable_drift_detection,
            'alerts_enabled': self.enable_alerts
        }
        
        if self.enable_monitoring:
            latest_metrics = self.metrics_collector.get_latest_metrics()
            summary['latest_metrics'] = latest_metrics
            summary['metrics_count'] = len(self.metrics_collector.metrics_history)
        
        if self.enable_drift_detection:
            drift_summary = self.drift_detector.get_drift_summary()
            summary['drift_summary'] = drift_summary
        
        if self.enable_alerts:
            alert_summary = self.alert_system.get_alert_summary()
            summary['alert_summary'] = alert_summary
        
        return summary
    
    def flush_metrics(self) -> None:
        """Flush any remaining buffered metrics"""
        if self.metrics_buffer:
            self._save_buffered_metrics()
            logger.info("All buffered metrics flushed")
    
    def get_metrics_report(self, limit: int = 100) -> Dict[str, Any]:
        """Get detailed metrics report"""
        if not self.enable_monitoring:
            return {}
        
        history = self.metrics_collector.get_metrics_history(limit=limit)
        
        metrics_names = ['precision_at_5', 'precision_at_10', 'recall_at_5', 'recall_at_10',
                        'f1_at_5', 'f1_at_10', 'mrr', 'ndcg_at_10', 'avg_response_time',
                        'avg_relevance_score']
        
        report = {}
        for metric_name in metrics_names:
            stats = self.metrics_collector.compute_metric_statistics(metric_name, limit=limit)
            report[metric_name] = stats
        
        return report
    
    def export_monitoring_data(self, export_dir: str = "monitoring_export") -> None:
        """Export all monitoring data"""
        Path(export_dir).mkdir(exist_ok=True)
        
        if self.enable_monitoring:
            self.metrics_collector.export_metrics(f"{export_dir}/metrics.json", format="json")
            self.metrics_collector.export_metrics(f"{export_dir}/metrics.csv", format="csv")
        
        if self.enable_drift_detection:
            self.drift_detector.export_drift_report(f"{export_dir}/drift_report.json", format="json")
            self.drift_detector.export_drift_report(f"{export_dir}/drift_report.csv", format="csv")
        
        if self.enable_alerts:
            self.alert_system.export_alerts(f"{export_dir}/alerts.json", format="json")
            self.alert_system.export_alerts(f"{export_dir}/alerts.csv", format="csv")
        
        logger.info(f"Monitoring data exported to {export_dir}")


# Example usage
if __name__ == "__main__":
    # Initialize RAG pipeline
    rag_pipeline = UniversalRAGPipeline()
    
    # Wrap with monitoring
    monitored_pipeline = MonitoredRAGPipeline(rag_pipeline)
    
    # Example queries
    test_queries = [
        ("What is the company onboarding process?", ["doc_1", "doc_2"]),
        ("How do we handle remote work?", ["doc_3"]),
        ("What are the benefits policies?", ["doc_4", "doc_5"])
    ]
    
    # Generate responses with monitoring
    for query, relevant_docs in test_queries:
        print(f"\nQuery: {query}")
        result = monitored_pipeline.generate_with_monitoring(
            query=query,
            relevant_doc_ids=relevant_docs
        )
        print(f"Response: {result['answer'][:100]}...")
    
    # Flush remaining metrics
    monitored_pipeline.flush_metrics()
    
    # Print summary
    print("\n=== Monitoring Summary ===")
    summary = monitored_pipeline.get_monitoring_summary()
    print(f"Metrics tracked: {summary['metrics_count']}")
    print(f"Alerts triggered: {summary['alert_summary']['total_alerts']}")
    print(f"Data drifts detected: {summary['drift_summary']['drift_count']}")
    
    # Export data
    monitored_pipeline.export_monitoring_data()
