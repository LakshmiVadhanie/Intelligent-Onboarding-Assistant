"""
Unit tests for monitoring system
Tests metrics collection, drift detection, and alert triggering
"""

import pytest
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.monitoring.metrics_collector import MetricsCollector, ModelMetrics
from src.monitoring.data_drift_detector import DataDriftDetector
from src.monitoring.alert_system import ThresholdAlertSystem, AlertThreshold


class TestMetricsCollector:
    """Tests for MetricsCollector"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    @pytest.fixture
    def collector(self, temp_dir):
        """Create MetricsCollector instance"""
        return MetricsCollector(metrics_dir=f"{temp_dir}/metrics")
    
    def test_collect_query_metrics(self, collector):
        """Test collecting metrics for a single query"""
        metrics = collector.collect_query_metrics(
            query="test query",
            retrieved_ids=["doc1", "doc2", "doc3"],
            relevant_ids=["doc1", "doc3"],
            response_time=0.5,
            relevance_scores=[0.95, 0.87, 0.65]
        )
        
        assert "timestamp" in metrics
        assert metrics["query"] == "test query"
        assert metrics["response_time"] == 0.5
        assert metrics["precision_at_5"] > 0
        assert metrics["retrieved_count"] == 3
        assert metrics["relevant_count"] == 2
    
    def test_save_and_load_metrics(self, collector):
        """Test saving and loading metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "precision_at_5": 0.75,
            "recall_at_5": 0.60,
            "f1_at_5": 0.67
        }
        
        collector.save_metrics(metrics)
        
        history = collector.get_metrics_history()
        assert len(history) > 0
        assert history[-1]["precision_at_5"] == 0.75
    
    def test_aggregate_metrics(self, collector):
        """Test aggregating multiple metrics"""
        metrics_list = [
            {
                "timestamp": datetime.now().isoformat(),
                "precision_at_5": 0.8,
                "recall_at_5": 0.7,
                "f1_at_5": 0.75,
                "mrr": 0.5,
                "ndcg_at_10": 0.6,
                "response_time": 0.5,
                "avg_relevance_score": 0.85
            },
            {
                "timestamp": datetime.now().isoformat(),
                "precision_at_5": 0.7,
                "recall_at_5": 0.65,
                "f1_at_5": 0.675,
                "mrr": 0.45,
                "ndcg_at_10": 0.55,
                "response_time": 0.6,
                "avg_relevance_score": 0.80
            }
        ]
        
        aggregated = collector.aggregate_metrics(metrics_list)
        
        assert isinstance(aggregated, ModelMetrics)
        assert aggregated.precision_at_5 == 0.75
        assert aggregated.query_count == 2
    
    def test_compute_statistics(self, collector):
        """Test computing metric statistics"""
        for i in range(10):
            metrics = {
                "timestamp": (datetime.now() - timedelta(seconds=i)).isoformat(),
                "precision_at_5": 0.5 + (i * 0.05),
                "recall_at_5": 0.4 + (i * 0.03)
            }
            collector.save_metrics(metrics)
        
        stats = collector.compute_metric_statistics("precision_at_5")
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["count"] == 10
        assert stats["min"] < stats["mean"] < stats["max"]


class TestDataDriftDetector:
    """Tests for DataDriftDetector"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    @pytest.fixture
    def detector(self, temp_dir):
        """Create DataDriftDetector instance"""
        return DataDriftDetector(
            baseline_dir=f"{temp_dir}/baseline",
            drift_reports_dir=f"{temp_dir}/drift"
        )
    
    def test_set_baseline(self, detector):
        """Test setting baseline distribution"""
        baseline_data = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        
        detector.set_baseline(baseline_data, feature_name="test_feature")
        
        assert "test_feature" in detector.baseline_stats
        baseline_info = detector.baseline_stats["test_feature"]
        assert baseline_info["mean"] == 0.7
        assert baseline_info["sample_count"] == 5
    
    def test_detect_statistical_drift(self, detector):
        """Test statistical drift detection"""
        baseline = np.random.normal(0.7, 0.1, 100)
        detector.set_baseline(baseline, feature_name="scores")
        
        # No drift case
        current_no_drift = np.random.normal(0.7, 0.1, 50)
        result = detector.detect_statistical_drift(
            current_no_drift,
            feature_name="scores"
        )
        
        # Should not detect drift (same distribution)
        assert result is not None
        assert isinstance(result.p_value, float)
    
    def test_detect_embedding_drift(self, detector):
        """Test embedding drift detection"""
        baseline = np.random.randn(100, 384)
        current = np.random.randn(50, 384)
        
        result = detector.detect_embedding_drift(baseline, current)
        
        assert result is not None
        assert 0 <= result.drift_score <= 1
        assert isinstance(result.is_drift_detected, bool)
    
    def test_save_drift_report(self, detector):
        """Test saving drift report"""
        baseline = np.random.normal(0.7, 0.1, 100)
        detector.set_baseline(baseline, feature_name="test")
        
        current = np.random.normal(0.2, 0.1, 50)
        result = detector.detect_statistical_drift(current, feature_name="test")
        
        detector.save_drift_report(result)
        
        summary = detector.get_drift_summary()
        assert summary["total_checks"] >= 1


class TestThresholdAlertSystem:
    """Tests for ThresholdAlertSystem"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    @pytest.fixture
    def alert_system(self, temp_dir):
        """Create ThresholdAlertSystem instance"""
        return ThresholdAlertSystem(alerts_dir=f"{temp_dir}/alerts")
    
    def test_set_threshold(self, alert_system):
        """Test setting alert threshold"""
        threshold = AlertThreshold(
            metric_name="test_metric",
            lower_threshold=0.5,
            upper_threshold=0.9,
            severity="critical"
        )
        
        alert_system.set_threshold(threshold)
        
        assert "test_metric" in alert_system.thresholds
        assert alert_system.thresholds["test_metric"].lower_threshold == 0.5
    
    def test_check_metric_below_threshold(self, alert_system):
        """Test detecting metric below threshold"""
        threshold = AlertThreshold(
            metric_name="precision",
            lower_threshold=0.5,
            severity="warning"
        )
        alert_system.set_threshold(threshold)
        
        # Value below threshold
        alert = alert_system.check_metric("precision", 0.3)
        
        assert alert is not None
        assert alert.metric_name == "precision"
        assert alert.current_value == 0.3
        assert alert.severity == "warning"
    
    def test_check_metric_above_threshold(self, alert_system):
        """Test detecting metric above threshold"""
        threshold = AlertThreshold(
            metric_name="response_time",
            upper_threshold=5.0,
            severity="critical"
        )
        alert_system.set_threshold(threshold)
        
        # Value above upper threshold
        alert = alert_system.check_metric("response_time", 6.0)
        
        assert alert is not None
        assert alert.severity == "critical"
    
    def test_check_metric_within_threshold(self, alert_system):
        """Test metric within acceptable range"""
        threshold = AlertThreshold(
            metric_name="precision",
            lower_threshold=0.5,
            severity="warning"
        )
        alert_system.set_threshold(threshold)
        
        # Value within threshold
        alert = alert_system.check_metric("precision", 0.7)
        
        assert alert is None
    
    def test_check_multiple_metrics(self, alert_system):
        """Test checking multiple metrics at once"""
        thresholds = [
            AlertThreshold("metric_1", 0.5, severity="warning"),
            AlertThreshold("metric_2", 0.6, severity="warning"),
            AlertThreshold("metric_3", 0.7, severity="warning")
        ]
        
        for t in thresholds:
            alert_system.set_threshold(t)
        
        metrics = {
            "metric_1": 0.4,  # Below threshold
            "metric_2": 0.8,  # OK
            "metric_3": 0.5   # Below threshold
        }
        
        alerts = alert_system.check_metrics(metrics)
        
        assert len(alerts) == 2
        assert any(a.metric_name == "metric_1" for a in alerts)
        assert any(a.metric_name == "metric_3" for a in alerts)
    
    def test_trigger_alert(self, alert_system):
        """Test triggering an alert"""
        threshold = AlertThreshold("precision", 0.5, severity="critical")
        alert_system.set_threshold(threshold)
        
        alert = alert_system.check_metric("precision", 0.3)
        alert_system.trigger_alert(alert)
        
        summary = alert_system.get_alert_summary()
        assert summary["total_alerts"] >= 1
    
    def test_custom_handler(self, alert_system):
        """Test registering custom alert handler"""
        handled_alerts = []
        
        def custom_handler(alert):
            handled_alerts.append(alert)
        
        alert_system.register_handler("custom", custom_handler)
        
        threshold = AlertThreshold("test", 0.5)
        alert_system.set_threshold(threshold)
        
        # Note: custom handlers are not called by trigger_alert
        # This tests registration
        assert len(alert_system.alert_handlers["custom"]) > 0


class TestIntegration:
    """Integration tests for complete monitoring workflow"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    def test_complete_monitoring_workflow(self, temp_dir):
        """Test complete monitoring workflow"""
        # Initialize components
        collector = MetricsCollector(metrics_dir=f"{temp_dir}/metrics")
        detector = DataDriftDetector(
            baseline_dir=f"{temp_dir}/baseline",
            drift_reports_dir=f"{temp_dir}/drift"
        )
        alert_system = ThresholdAlertSystem(alerts_dir=f"{temp_dir}/alerts")
        
        # Set baseline for drift detection
        baseline = np.random.normal(0.7, 0.1, 100)
        detector.set_baseline(baseline, feature_name="relevance")
        
        # Set alert thresholds
        alert_system.set_threshold(
            AlertThreshold("precision_at_5", 0.5, severity="warning")
        )
        
        # Simulate queries
        for i in range(5):
            # Collect metrics
            metrics = collector.collect_query_metrics(
                query=f"test query {i}",
                retrieved_ids=["doc1", "doc2"],
                relevant_ids=["doc1"],
                response_time=0.5,
                relevance_scores=[0.9, 0.7]
            )
            
            collector.save_metrics(metrics)
            
            # Check for alerts
            alert = alert_system.check_metric("precision_at_5", metrics.get("precision_at_5", 0.5))
            
            # Detect drift
            current = np.random.normal(0.7, 0.1, 10)
            drift_result = detector.detect_statistical_drift(
                current,
                feature_name="relevance"
            )
            
            if drift_result and drift_result.is_drift_detected:
                detector.save_drift_report(drift_result)
        
        # Verify data was collected
        metrics_history = collector.get_metrics_history()
        assert len(metrics_history) == 5
        
        drift_summary = detector.get_drift_summary()
        assert drift_summary["total_checks"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
