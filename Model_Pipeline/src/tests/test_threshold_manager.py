"""
Unit tests for Step 3: Threshold Management & Automated Retraining Triggering
"""

import json
import pytest
from pathlib import Path
from datetime import datetime
from src.monitoring.threshold_manager import (
    ThresholdManager,
    ThresholdConfig,
    RetrainingTrigger,
    ThresholdViolation
)


class TestThresholdConfig:
    """Test threshold configuration"""
    
    def test_default_config(self):
        """Test default threshold values"""
        config = ThresholdConfig()
        
        assert config.precision_at_5 == 0.5
        assert config.overall_drift_rate == 0.5
        assert config.critical_threshold == 0.8
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ThresholdConfig(
            precision_at_5=0.6,
            overall_drift_rate=0.7
        )
        
        assert config.precision_at_5 == 0.6
        assert config.overall_drift_rate == 0.7
    
    def test_config_to_dict(self):
        """Test config serialization"""
        config = ThresholdConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'precision_at_5' in config_dict
        assert config_dict['precision_at_5'] == 0.5


class TestThresholdManager:
    """Test threshold management"""
    
    @pytest.fixture
    def manager(self):
        """Create threshold manager"""
        config = ThresholdConfig()
        return ThresholdManager(config=config)
    
    def test_manager_initialization(self, manager):
        """Test manager initialization"""
        assert manager.config is not None
        assert manager.history_dir.exists()
        assert isinstance(manager.violation_history, list)
    
    def test_performance_threshold_normal(self, manager):
        """Test normal performance metrics"""
        metrics = {
            'precision_at_5': 0.65,
            'recall_at_5': 0.55,
            'f1_at_5': 0.60,
            'mrr': 0.50,
            'ndcg_at_10': 0.55,
            'relevance_score': 0.72,
            'response_time': 2.1
        }
        
        violations = manager.check_performance_thresholds(metrics)
        assert len(violations) == 0
    
    def test_performance_threshold_violations(self, manager):
        """Test performance degradation detection"""
        metrics = {
            'precision_at_5': 0.35,
            'recall_at_5': 0.25,
            'f1_at_5': 0.28,
            'mrr': 0.28,
            'ndcg_at_10': 0.35,
            'relevance_score': 0.42,
            'response_time': 8.2
        }
        
        violations = manager.check_performance_thresholds(metrics)
        
        assert len(violations) == 7
        assert all(v.should_trigger_retraining for v in violations 
                  if v.severity in ['high', 'critical'])
    
    def test_violation_structure(self, manager):
        """Test violation object structure"""
        metrics = {'precision_at_5': 0.3}
        violations = manager.check_performance_thresholds(metrics)
        
        assert len(violations) > 0
        v = violations[0]
        
        assert hasattr(v, 'timestamp')
        assert hasattr(v, 'metric_name')
        assert hasattr(v, 'metric_value')
        assert hasattr(v, 'threshold_value')
        assert hasattr(v, 'severity')
        assert hasattr(v, 'should_trigger_retraining')
        assert hasattr(v, 'recommended_action')
    
    def test_drift_threshold_normal(self, manager):
        """Test normal drift detection"""
        drift_report = {
            'drift_rate': 0.15,
            'total_features': 5,
            'drifted_features': {'query_length': {}},
            'anomaly_rate': 0.002
        }
        
        violations = manager.check_drift_thresholds(drift_report)
        assert len(violations) == 0
    
    def test_drift_threshold_critical(self, manager):
        """Test critical drift detection"""
        drift_report = {
            'drift_rate': 0.80,
            'total_features': 5,
            'drifted_features': {
                'query_length': {},
                'document_count': {},
                'relevance_score': {},
                'response_time': {}
            },
            'anomaly_rate': 0.008
        }
        
        violations = manager.check_drift_thresholds(drift_report)
        
        assert len(violations) > 0
        assert any(v.severity == 'critical' for v in violations)
    
    def test_drift_feature_percentage(self, manager):
        """Test feature drift percentage checking"""
        drift_report = {
            'drift_rate': 0.35,  # Below overall threshold
            'total_features': 5,
            'drifted_features': {
                'query_length': {},
                'document_count': {},
                'relevance_score': {},  # 60% of features
            },
            'anomaly_rate': 0.001
        }
        
        violations = manager.check_drift_thresholds(drift_report)
        
        # Should detect feature drift percentage violation
        assert any(v.metric_name == 'feature_drift_count' for v in violations)
    
    def test_retraining_decision_no_violations(self, manager):
        """Test retraining decision with no violations"""
        decision = manager.evaluate_retraining_need([], [])
        
        assert decision['should_retrain'] is False
        assert decision['priority'] == 0
        assert decision['violations'] == 0
    
    def test_retraining_decision_critical_violations(self, manager):
        """Test retraining decision with critical violations"""
        metrics = {
            'precision_at_5': 0.2,
            'recall_at_5': 0.1,
            'f1_at_5': 0.15
        }
        
        perf_violations = manager.check_performance_thresholds(metrics)
        decision = manager.evaluate_retraining_need(perf_violations, [])
        
        assert decision['should_retrain'] is True
        assert decision['priority'] >= 3
    
    def test_retraining_decision_multiple_high_violations(self, manager):
        """Test retraining with multiple high violations"""
        metrics = {
            'precision_at_5': 0.4,
            'recall_at_5': 0.25,
            'response_time': 7.0
        }
        
        perf_violations = manager.check_performance_thresholds(metrics)
        
        high_violations = [v for v in perf_violations if v.severity == 'high']
        assert len(high_violations) >= 2
        
        decision = manager.evaluate_retraining_need(perf_violations, [])
        assert decision['should_retrain'] is True
        assert decision['priority'] >= 4
    
    def test_violation_percentage_calculation(self, manager):
        """Test violation percentage calculation"""
        metrics = {'precision_at_5': 0.35}  # 30% below 0.5 threshold
        violations = manager.check_performance_thresholds(metrics)
        
        assert len(violations) > 0
        assert violations[0].violation_percentage > 20
    
    def test_get_retraining_urgency(self, manager):
        """Test urgency check"""
        metrics = {'precision_at_5': 0.2}
        manager.check_performance_thresholds(metrics)
        
        should_retrain, priority, reason = manager.get_retraining_urgency()
        
        assert isinstance(should_retrain, bool)
        assert isinstance(priority, int)
        assert isinstance(reason, str)
    
    def test_violation_summary(self, manager):
        """Test violation summary"""
        metrics = {
            'precision_at_5': 0.3,
            'recall_at_5': 0.2,
            'f1_at_5': 0.25
        }
        manager.check_performance_thresholds(metrics)
        
        summary = manager.get_violation_summary(hours=24)
        
        assert 'total_violations' in summary
        assert 'by_severity' in summary
        assert summary['total_violations'] > 0
    
    def test_threshold_update(self, manager):
        """Test dynamic threshold updates"""
        original_precision = manager.config.precision_at_5
        
        manager.update_thresholds({'precision_at_5': 0.45})
        
        assert manager.config.precision_at_5 == 0.45
        assert manager.config.precision_at_5 != original_precision
    
    def test_export_configuration(self, manager):
        """Test configuration export"""
        filepath = "tests/tmp_config.json"
        manager.export_configuration(filepath)
        
        assert Path(filepath).exists()
        
        with open(filepath) as f:
            exported = json.load(f)
        
        assert exported['precision_at_5'] == manager.config.precision_at_5
        
        # Cleanup
        Path(filepath).unlink()
    
    def test_import_configuration(self, manager):
        """Test configuration import"""
        # Create a config file
        test_config = {
            'precision_at_5': 0.6,
            'overall_drift_rate': 0.7
        }
        filepath = "tests/tmp_import_config.json"
        with open(filepath, 'w') as f:
            json.dump(test_config, f)
        
        # Import it
        manager.import_configuration(filepath)
        
        assert manager.config.precision_at_5 == 0.6
        assert manager.config.overall_drift_rate == 0.7
        
        # Cleanup
        Path(filepath).unlink()


class TestRetrainingTrigger:
    """Test retraining trigger"""
    
    @pytest.fixture
    def trigger(self):
        """Create retraining trigger"""
        return RetrainingTrigger(ci_cd_pipeline_name="test_pipeline")
    
    def test_trigger_initialization(self, trigger):
        """Test trigger initialization"""
        assert trigger.ci_cd_pipeline_name == "test_pipeline"
        assert trigger.trigger_dir.exists()
    
    def test_trigger_retraining(self, trigger):
        """Test triggering retraining"""
        decision = {
            'should_retrain': True,
            'priority': 5,
            'reason': 'Critical violations',
            'violations': 3,
            'critical_violations': 2,
            'high_violations': 1,
            'actions': ['Retrain model']
        }
        
        result = trigger.trigger_retraining(decision)
        
        assert result['triggered'] is True
        assert 'trigger_id' in result
        assert result['pipeline'] == "test_pipeline"
        assert result['priority'] == 5
    
    def test_no_trigger_when_not_needed(self, trigger):
        """Test no trigger when not needed"""
        decision = {
            'should_retrain': False,
            'reason': 'Thresholds not violated'
        }
        
        result = trigger.trigger_retraining(decision)
        
        assert result['triggered'] is False
    
    def test_trigger_with_metadata(self, trigger):
        """Test trigger with metadata"""
        decision = {
            'should_retrain': True,
            'priority': 4,
            'reason': 'Test',
            'violations': 2,
            'critical_violations': 0,
            'high_violations': 2,
            'actions': []
        }
        
        metadata = {
            'data_shift_detected': True,
            'drift_rate': 0.75
        }
        
        result = trigger.trigger_retraining(decision, metadata=metadata)
        
        assert result['triggered'] is True
    
    def test_trigger_history(self, trigger):
        """Test trigger history"""
        decision = {
            'should_retrain': True,
            'priority': 3,
            'reason': 'Test',
            'violations': 1,
            'critical_violations': 0,
            'high_violations': 1,
            'actions': []
        }
        
        result1 = trigger.trigger_retraining(decision)
        trigger_id_1 = result1['trigger_id']
        
        recent = trigger.get_recent_triggers(limit=10)
        
        assert len(recent) > 0
        assert any(t['trigger_id'] == trigger_id_1 for t in recent)
    
    def test_get_trigger_status(self, trigger):
        """Test getting trigger status"""
        decision = {
            'should_retrain': True,
            'priority': 2,
            'reason': 'Test',
            'violations': 1,
            'critical_violations': 0,
            'high_violations': 1,
            'actions': []
        }
        
        result = trigger.trigger_retraining(decision)
        trigger_id = result['trigger_id']
        
        status = trigger.get_trigger_status(trigger_id)
        
        assert status is not None
        assert status['trigger_id'] == trigger_id


class TestIntegration:
    """Integration tests for Step 3"""
    
    def test_threshold_to_retraining_workflow(self):
        """Test complete workflow from thresholds to retraining"""
        manager = ThresholdManager(config=ThresholdConfig())
        trigger = RetrainingTrigger()
        
        # Simulate degraded performance + drift
        metrics = {
            'precision_at_5': 0.35,
            'recall_at_5': 0.25,
            'response_time': 8.0
        }
        
        drift_report = {
            'drift_rate': 0.75,
            'total_features': 5,
            'drifted_features': {
                'query_length': {},
                'document_count': {},
                'relevance_score': {},
                'response_time': {}
            },
            'anomaly_rate': 0.008
        }
        
        # Check thresholds
        perf_violations = manager.check_performance_thresholds(metrics)
        drift_violations = manager.check_drift_thresholds(drift_report)
        
        assert len(perf_violations) > 0
        assert len(drift_violations) > 0
        
        # Evaluate retraining
        decision = manager.evaluate_retraining_need(perf_violations, drift_violations)
        
        assert decision['should_retrain'] is True
        assert decision['priority'] == 5
        
        # Trigger retraining
        result = trigger.trigger_retraining(decision)
        
        assert result['triggered'] is True
        assert result['priority'] == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
