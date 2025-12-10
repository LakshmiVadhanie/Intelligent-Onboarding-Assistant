"""
Step 3: Threshold Management & Automated Retraining Triggering

Manages predefined thresholds for model performance and data drift.
Automatically triggers retraining when thresholds are exceeded.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configuration for performance and drift thresholds"""
    # Performance metrics thresholds
    precision_at_5: float = 0.5          # Minimum acceptable precision
    recall_at_5: float = 0.3             # Minimum acceptable recall
    f1_at_5: float = 0.35                # Minimum acceptable F1 score
    mrr: float = 0.3                     # Mean Reciprocal Rank minimum
    ndcg_at_10: float = 0.4              # NDCG@10 minimum
    relevance_score: float = 0.5         # Minimum relevance score
    response_time: float = 5.0           # Maximum response time (seconds)
    
    # Data drift thresholds
    overall_drift_rate: float = 0.5      # Overall drift rate threshold (50%)
    feature_drift_score: float = 0.6     # Individual feature drift score (0-1)
    feature_drift_percentage: int = 40   # % of features that can drift
    data_anomaly_rate: float = 0.05      # Maximum anomaly rate (5%)
    
    # Severity-based thresholds
    critical_threshold: float = 0.8      # >80% drift = critical
    high_threshold: float = 0.5          # 50-80% drift = high
    medium_threshold: float = 0.2        # 20-50% drift = medium
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ThresholdViolation:
    """Represents a threshold violation event"""
    timestamp: str
    metric_name: str
    metric_value: float
    threshold_value: float
    severity: str  # 'info', 'warning', 'high', 'critical'
    violation_percentage: float  # How much over threshold
    should_trigger_retraining: bool
    retraining_priority: int  # 1-5, where 5 is highest
    recommended_action: str


class ThresholdManager:
    """
    Manages performance and drift thresholds.
    Determines when to trigger retraining based on violations.
    """
    
    def __init__(self, 
                 config: Optional[ThresholdConfig] = None,
                 history_dir: str = "experiments/threshold_history"):
        """
        Initialize Threshold Manager
        
        Args:
            config: ThresholdConfig object with custom thresholds
            history_dir: Directory to store violation history
        """
        self.config = config or ThresholdConfig()
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        self.violation_history: List[ThresholdViolation] = []
        self.retraining_decisions: List[Dict[str, Any]] = []
        
        self._load_history()
        logger.info("ThresholdManager initialized")
    
    def _load_history(self) -> None:
        """Load violation history from disk"""
        history_file = self.history_dir / "violation_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.violation_history = data.get('violations', [])
                logger.info(f"Loaded {len(self.violation_history)} historical violations")
            except Exception as e:
                logger.error(f"Error loading violation history: {e}")
    
    def _save_history(self) -> None:
        """Save violation history to disk"""
        try:
            history_file = self.history_dir / "violation_history.json"
            with open(history_file, 'w') as f:
                json.dump({
                    'violations': self.violation_history,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving violation history: {e}")
    
    def check_performance_thresholds(self, 
                                    metrics: Dict[str, float]) -> List[ThresholdViolation]:
        """
        Check if performance metrics violate thresholds
        
        Args:
            metrics: Dictionary of metric_name -> value
            
        Returns:
            List of threshold violations
        """
        violations = []
        
        # Define metric checks
        checks = [
            ('precision_at_5', self.config.precision_at_5, 'below'),
            ('recall_at_5', self.config.recall_at_5, 'below'),
            ('f1_at_5', self.config.f1_at_5, 'below'),
            ('mrr', self.config.mrr, 'below'),
            ('ndcg_at_10', self.config.ndcg_at_10, 'below'),
            ('relevance_score', self.config.relevance_score, 'below'),
            ('response_time', self.config.response_time, 'above'),
        ]
        
        for metric_name, threshold, direction in checks:
            if metric_name not in metrics:
                continue
            
            value = metrics[metric_name]
            
            # Check violation
            violated = False
            if direction == 'below' and value < threshold:
                violated = True
                violation_pct = ((threshold - value) / threshold) * 100
            elif direction == 'above' and value > threshold:
                violated = True
                violation_pct = ((value - threshold) / threshold) * 100
            else:
                continue
            
            # Determine severity
            severity = self._determine_severity_performance(metric_name, value, threshold)
            should_retrain = severity in ['high', 'critical']
            priority = 5 if severity == 'critical' else 3 if severity == 'high' else 1
            
            violation = ThresholdViolation(
                timestamp=datetime.now().isoformat(),
                metric_name=metric_name,
                metric_value=value,
                threshold_value=threshold,
                severity=severity,
                violation_percentage=violation_pct,
                should_trigger_retraining=should_retrain,
                retraining_priority=priority,
                recommended_action=self._get_action_recommendation(
                    metric_name, severity, value, threshold
                )
            )
            
            violations.append(violation)
            self.violation_history.append(asdict(violation))
        
        if violations:
            self._save_history()
            logger.warning(f"Found {len(violations)} performance threshold violations")
        
        return violations
    
    def check_drift_thresholds(self,
                              drift_report: Dict[str, Any]) -> List[ThresholdViolation]:
        """
        Check if data drift violates thresholds
        
        Args:
            drift_report: Data shift detection report
            
        Returns:
            List of threshold violations
        """
        violations = []
        
        # Overall drift rate check
        overall_drift = drift_report.get('drift_rate', 0.0)
        if overall_drift > self.config.overall_drift_rate:
            violation_pct = ((overall_drift - self.config.overall_drift_rate) / 
                           self.config.overall_drift_rate) * 100
            
            severity = self._determine_severity_drift(overall_drift)
            should_retrain = severity in ['high', 'critical']
            priority = 5 if severity == 'critical' else 3 if severity == 'high' else 2
            
            violation = ThresholdViolation(
                timestamp=datetime.now().isoformat(),
                metric_name='overall_drift_rate',
                metric_value=overall_drift,
                threshold_value=self.config.overall_drift_rate,
                severity=severity,
                violation_percentage=violation_pct,
                should_trigger_retraining=should_retrain,
                retraining_priority=priority,
                recommended_action=self._get_drift_recommendation(severity, overall_drift)
            )
            violations.append(violation)
        
        # Feature drift check
        drifted_features = drift_report.get('drifted_features', {})
        total_features = drift_report.get('total_features', 1)
        
        if total_features > 0:
            drifted_count = len(drifted_features)
            drifted_percentage = (drifted_count / total_features) * 100
            
            if drifted_percentage > self.config.feature_drift_percentage:
                violation_pct = drifted_percentage - self.config.feature_drift_percentage
                
                severity = 'high' if drifted_percentage > 60 else 'medium'
                should_retrain = severity == 'high'
                
                violation = ThresholdViolation(
                    timestamp=datetime.now().isoformat(),
                    metric_name='feature_drift_count',
                    metric_value=drifted_percentage,
                    threshold_value=self.config.feature_drift_percentage,
                    severity=severity,
                    violation_percentage=violation_pct,
                    should_trigger_retraining=should_retrain,
                    retraining_priority=3,
                    recommended_action=f"Retrain with focus on {drifted_count}/{total_features} drifted features"
                )
                violations.append(violation)
        
        # Data quality check
        anomaly_rate = drift_report.get('anomaly_rate', 0.0)
        if anomaly_rate > self.config.data_anomaly_rate:
            violation = ThresholdViolation(
                timestamp=datetime.now().isoformat(),
                metric_name='data_anomaly_rate',
                metric_value=anomaly_rate,
                threshold_value=self.config.data_anomaly_rate,
                severity='warning',
                violation_percentage=((anomaly_rate - self.config.data_anomaly_rate) / 
                                    self.config.data_anomaly_rate) * 100,
                should_trigger_retraining=False,
                retraining_priority=1,
                recommended_action="Monitor data quality pipeline"
            )
            violations.append(violation)
        
        if violations:
            self._save_history()
            logger.warning(f"Found {len(violations)} drift threshold violations")
        
        return violations
    
    def _determine_severity_performance(self, metric_name: str, 
                                       value: float, threshold: float) -> str:
        """Determine violation severity for performance metrics"""
        if metric_name == 'response_time':
            if value > threshold * 2:
                return 'critical'
            elif value > threshold * 1.5:
                return 'high'
            return 'warning'
        else:
            diff_pct = abs(value - threshold) / threshold
            if diff_pct > 0.3:
                return 'critical'
            elif diff_pct > 0.15:
                return 'high'
            return 'warning'
    
    def _determine_severity_drift(self, drift_rate: float) -> str:
        """Determine violation severity for drift"""
        if drift_rate >= self.config.critical_threshold:
            return 'critical'
        elif drift_rate >= self.config.high_threshold:
            return 'high'
        elif drift_rate >= self.config.medium_threshold:
            return 'medium'
        return 'warning'
    
    def _get_action_recommendation(self, metric_name: str, severity: str,
                                  value: float, threshold: float) -> str:
        """Get recommended action for performance violation"""
        recommendations = {
            'precision_at_5': "Improve retrieval relevance ranking",
            'recall_at_5': "Expand retrieval scope or improve keyword matching",
            'f1_at_5': "Balance precision and recall - reweight features",
            'mrr': "Improve document ranking for top results",
            'ndcg_at_10': "Enhance ranking order effectiveness",
            'relevance_score': "Review query-document similarity metrics",
            'response_time': "Optimize vector search or reduce dataset size"
        }
        
        base_action = recommendations.get(metric_name, "Investigate metric")
        
        if severity == 'critical':
            return f"URGENT: {base_action}. Trigger immediate retraining."
        elif severity == 'high':
            return f"Investigate {metric_name}. {base_action}. Schedule retraining."
        else:
            return f"Monitor {metric_name}. {base_action}."
    
    def _get_drift_recommendation(self, severity: str, drift_rate: float) -> str:
        """Get recommended action for drift violation"""
        if severity == 'critical':
            return f"CRITICAL: {drift_rate:.1%} data drift detected. Retrain immediately with recent data."
        elif severity == 'high':
            return f"HIGH: {drift_rate:.1%} data drift detected. Schedule retraining soon."
        else:
            return f"MEDIUM: {drift_rate:.1%} data drift detected. Plan retraining."
    
    def evaluate_retraining_need(self,
                                performance_violations: List[ThresholdViolation],
                                drift_violations: List[ThresholdViolation]) -> Dict[str, Any]:
        """
        Evaluate overall need for retraining based on all violations
        
        Args:
            performance_violations: List of performance threshold violations
            drift_violations: List of drift threshold violations
            
        Returns:
            Dictionary with retraining decision info
        """
        all_violations = performance_violations + drift_violations
        
        if not all_violations:
            decision = {
                'should_retrain': False,
                'priority': 0,
                'reason': 'All metrics within acceptable thresholds',
                'violations': 0,
                'critical_violations': 0,
                'high_violations': 0,
                'actions': []
            }
            return decision
        
        critical_count = sum(1 for v in all_violations if v.severity == 'critical')
        high_count = sum(1 for v in all_violations if v.severity == 'high')
        
        # Decision logic
        should_retrain = False
        priority = 0
        reason = ""
        
        if critical_count > 0:
            should_retrain = True
            priority = 5
            reason = f"CRITICAL: {critical_count} critical violations detected"
        elif high_count >= 2:
            should_retrain = True
            priority = 4
            reason = f"HIGH: {high_count} high-severity violations detected"
        elif high_count == 1:
            should_retrain = True
            priority = 3
            reason = f"MEDIUM-HIGH: {high_count} high-severity violation detected"
        elif len(all_violations) >= 3:
            should_retrain = True
            priority = 2
            reason = f"ELEVATED: {len(all_violations)} violations across multiple metrics"
        
        # Extract actions
        actions = list(set(v.recommended_action for v in all_violations))
        
        decision = {
            'should_retrain': should_retrain,
            'priority': priority,
            'reason': reason,
            'violations': len(all_violations),
            'critical_violations': critical_count,
            'high_violations': high_count,
            'actions': actions,
            'timestamp': datetime.now().isoformat()
        }
        
        self.retraining_decisions.append(decision)
        self._save_retraining_decisions()
        
        logger.info(f"Retraining decision: {should_retrain} (priority={priority})")
        
        return decision
    
    def _save_retraining_decisions(self) -> None:
        """Save retraining decisions to disk"""
        try:
            decisions_file = self.history_dir / "retraining_decisions.json"
            with open(decisions_file, 'w') as f:
                json.dump(self.retraining_decisions, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving retraining decisions: {e}")
    
    def get_retraining_urgency(self) -> Tuple[bool, int, str]:
        """
        Quick check for retraining urgency
        
        Returns:
            Tuple of (should_retrain, priority_1_5, reason)
        """
        if not self.retraining_decisions:
            return False, 0, "No violations recorded"
        
        latest = self.retraining_decisions[-1]
        return (
            latest['should_retrain'],
            latest['priority'],
            latest['reason']
        )
    
    def get_violation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of violations in the last N hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Summary statistics
        """
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_violations = [
            v for v in self.violation_history
            if datetime.fromisoformat(v['timestamp']) > cutoff_time
        ]
        
        severity_counts = {}
        for v in recent_violations:
            severity = v['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_violations': len(recent_violations),
            'by_severity': severity_counts,
            'unique_metrics': len(set(v['metric_name'] for v in recent_violations)),
            'time_period_hours': hours,
            'latest_violation': recent_violations[-1] if recent_violations else None
        }
    
    def update_thresholds(self, new_config: Dict[str, float]) -> None:
        """
        Update threshold configuration
        
        Args:
            new_config: Dictionary of threshold_name -> value
        """
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated threshold {key} to {value}")
            else:
                logger.warning(f"Unknown threshold parameter: {key}")
    
    def export_configuration(self, filepath: str) -> None:
        """Export current configuration to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            logger.info(f"Configuration exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
    
    def import_configuration(self, filepath: str) -> None:
        """Import configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            self.update_thresholds(config_dict)
            logger.info(f"Configuration imported from {filepath}")
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")


class RetrainingTrigger:
    """
    Handles automated retraining triggering based on threshold violations
    """
    
    def __init__(self, 
                 ci_cd_pipeline_name: str = "model_retraining",
                 trigger_dir: str = "experiments/retraining_triggers"):
        """
        Initialize Retraining Trigger
        
        Args:
            ci_cd_pipeline_name: Name of CI/CD pipeline to trigger
            trigger_dir: Directory for trigger records
        """
        self.ci_cd_pipeline_name = ci_cd_pipeline_name
        self.trigger_dir = Path(trigger_dir)
        self.trigger_dir.mkdir(parents=True, exist_ok=True)
        
        self.trigger_history: List[Dict[str, Any]] = []
        self._load_trigger_history()
    
    def _load_trigger_history(self) -> None:
        """Load trigger history"""
        history_file = self.trigger_dir / "trigger_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.trigger_history = json.load(f)
            except Exception as e:
                logger.error(f"Error loading trigger history: {e}")
    
    def _save_trigger_history(self) -> None:
        """Save trigger history"""
        try:
            history_file = self.trigger_dir / "trigger_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.trigger_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving trigger history: {e}")
    
    def trigger_retraining(self,
                          decision_info: Dict[str, Any],
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Trigger retraining pipeline
        
        Args:
            decision_info: Retraining decision information
            metadata: Additional metadata for the trigger
            
        Returns:
            Trigger result information
        """
        if not decision_info.get('should_retrain'):
            logger.info("Retraining not needed")
            return {'triggered': False, 'reason': 'Thresholds not violated'}
        
        trigger_record = {
            'timestamp': datetime.now().isoformat(),
            'trigger_id': f"retrain_{int(datetime.now().timestamp() * 1000)}",
            'priority': decision_info.get('priority', 0),
            'reason': decision_info.get('reason', ''),
            'violations': decision_info.get('violations', 0),
            'critical_violations': decision_info.get('critical_violations', 0),
            'pipeline': self.ci_cd_pipeline_name,
            'status': 'triggered',
            'metadata': metadata or {}
        }
        
        self.trigger_history.append(trigger_record)
        self._save_trigger_history()
        
        logger.info(f"Retraining triggered: {trigger_record['trigger_id']} "
                   f"(priority={trigger_record['priority']})")
        
        return {
            'triggered': True,
            'trigger_id': trigger_record['trigger_id'],
            'pipeline': self.ci_cd_pipeline_name,
            'priority': trigger_record['priority'],
            'message': f"Retraining pipeline '{self.ci_cd_pipeline_name}' triggered"
        }
    
    def get_trigger_status(self, trigger_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific trigger"""
        for record in self.trigger_history:
            if record['trigger_id'] == trigger_id:
                return record
        return None
    
    def get_recent_triggers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trigger records"""
        return self.trigger_history[-limit:]
