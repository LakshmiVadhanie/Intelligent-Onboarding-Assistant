"""
Alert System for Model Monitoring
Triggers alerts when metrics fall below thresholds
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AlertThreshold:
    """Data class for alert thresholds"""
    metric_name: str
    lower_threshold: float
    upper_threshold: Optional[float] = None
    severity: str = "warning"  # 'info', 'warning', 'critical'
    enabled: bool = True
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Alert:
    """Data class for triggered alerts"""
    timestamp: str
    alert_id: str
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    message: str
    acknowledged: bool = False
    
    def to_dict(self):
        return asdict(self)


class ThresholdAlertSystem:
    """Manages alert thresholds and triggers"""
    
    def __init__(self,
                 alerts_dir: str = "experiments/monitoring/alerts",
                 email_enabled: bool = False,
                 slack_enabled: bool = False):
        """
        Initialize ThresholdAlertSystem
        
        Args:
            alerts_dir: Directory to store alerts
            email_enabled: Whether to send email alerts
            slack_enabled: Whether to send Slack alerts
        """
        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        
        self.email_enabled = email_enabled
        self.slack_enabled = slack_enabled
        
        self.thresholds = self._load_thresholds()
        self.alerts_history = self._load_alerts_history()
        self.alert_handlers: Dict[str, List[Callable]] = {
            "email": [],
            "slack": [],
            "webhook": [],
            "log": []
        }
        
        # Register default handler
        self._register_default_handlers()
    
    def _load_thresholds(self) -> Dict[str, AlertThreshold]:
        """Load thresholds from file"""
        thresholds_file = self.alerts_dir / "thresholds.json"
        if thresholds_file.exists():
            try:
                with open(thresholds_file, 'r') as f:
                    data = json.load(f)
                    return {
                        k: AlertThreshold(**v) for k, v in data.items()
                    }
            except Exception as e:
                logger.error(f"Error loading thresholds: {e}")
        return self._get_default_thresholds()
    
    def _get_default_thresholds(self) -> Dict[str, AlertThreshold]:
        """Get default thresholds for common metrics"""
        return {
            "precision_at_5": AlertThreshold(
                metric_name="precision_at_5",
                lower_threshold=0.5,
                severity="warning"
            ),
            "precision_at_10": AlertThreshold(
                metric_name="precision_at_10",
                lower_threshold=0.4,
                severity="warning"
            ),
            "recall_at_5": AlertThreshold(
                metric_name="recall_at_5",
                lower_threshold=0.3,
                severity="warning"
            ),
            "recall_at_10": AlertThreshold(
                metric_name="recall_at_10",
                lower_threshold=0.4,
                severity="warning"
            ),
            "f1_at_5": AlertThreshold(
                metric_name="f1_at_5",
                lower_threshold=0.35,
                severity="warning"
            ),
            "f1_at_10": AlertThreshold(
                metric_name="f1_at_10",
                lower_threshold=0.35,
                severity="warning"
            ),
            "mrr": AlertThreshold(
                metric_name="mrr",
                lower_threshold=0.3,
                severity="warning"
            ),
            "ndcg_at_10": AlertThreshold(
                metric_name="ndcg_at_10",
                lower_threshold=0.4,
                severity="warning"
            ),
            "avg_response_time": AlertThreshold(
                metric_name="avg_response_time",
                lower_threshold=0,
                upper_threshold=5.0,  # 5 seconds max
                severity="critical"
            ),
            "avg_relevance_score": AlertThreshold(
                metric_name="avg_relevance_score",
                lower_threshold=0.5,
                severity="warning"
            )
        }
    
    def _load_alerts_history(self) -> List[Dict[str, Any]]:
        """Load alerts history"""
        history_file = self.alerts_dir / "alerts_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading alerts history: {e}")
        return []
    
    def set_threshold(self, threshold: AlertThreshold) -> None:
        """Set alert threshold for a metric"""
        self.thresholds[threshold.metric_name] = threshold
        self._save_thresholds()
        logger.info(f"Threshold set for {threshold.metric_name}")
    
    def _save_thresholds(self) -> None:
        """Save thresholds to file"""
        try:
            thresholds_file = self.alerts_dir / "thresholds.json"
            data = {k: v.to_dict() for k, v in self.thresholds.items()}
            with open(thresholds_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving thresholds: {e}")
    
    def check_metric(self, metric_name: str, value: float) -> Optional[Alert]:
        """
        Check if metric value triggers an alert
        
        Args:
            metric_name: Name of the metric
            value: Current metric value
            
        Returns:
            Alert object if threshold breached, None otherwise
        """
        if metric_name not in self.thresholds:
            return None
        
        threshold = self.thresholds[metric_name]
        
        if not threshold.enabled:
            return None
        
        # Check if value violates threshold
        is_breach = False
        message = ""
        
        if value < threshold.lower_threshold:
            is_breach = True
            message = f"{metric_name} dropped below threshold: {value:.4f} < {threshold.lower_threshold:.4f}"
        
        if threshold.upper_threshold and value > threshold.upper_threshold:
            is_breach = True
            message = f"{metric_name} exceeded threshold: {value:.4f} > {threshold.upper_threshold:.4f}"
        
        if not is_breach:
            return None
        
        # Create alert
        alert = Alert(
            timestamp=datetime.now().isoformat(),
            alert_id=f"{metric_name}_{datetime.now().timestamp()}",
            metric_name=metric_name,
            current_value=value,
            threshold=threshold.lower_threshold,
            severity=threshold.severity,
            message=message
        )
        
        return alert
    
    def check_metrics(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        Check multiple metrics and return triggered alerts
        
        Args:
            metrics: Dictionary of metric_name -> value
            
        Returns:
            List of Alert objects
        """
        alerts = []
        for metric_name, value in metrics.items():
            alert = self.check_metric(metric_name, value)
            if alert:
                alerts.append(alert)
        return alerts
    
    def trigger_alert(self, alert: Alert) -> None:
        """
        Trigger an alert
        
        Args:
            alert: Alert object to trigger
        """
        # Save to history
        self.alerts_history.append(alert.to_dict())
        
        try:
            history_file = self.alerts_dir / "alerts_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.alerts_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
        
        # Call registered handlers
        logger.warning(f"ALERT TRIGGERED: {alert.message} (Severity: {alert.severity})")
        
        for handler in self.alert_handlers.get("log", []):
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in log handler: {e}")
        
        if self.email_enabled:
            for handler in self.alert_handlers.get("email", []):
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in email handler: {e}")
        
        if self.slack_enabled:
            for handler in self.alert_handlers.get("slack", []):
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in Slack handler: {e}")
        
        for handler in self.alert_handlers.get("webhook", []):
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in webhook handler: {e}")
    
    def _register_default_handlers(self) -> None:
        """Register default alert handlers"""
        # Log handler
        def log_handler(alert: Alert):
            logger.warning(
                f"[{alert.severity.upper()}] {alert.metric_name}: {alert.message}"
            )
        
        self.register_handler("log", log_handler)
    
    def register_handler(self, handler_type: str, handler: Callable) -> None:
        """
        Register a custom alert handler
        
        Args:
            handler_type: Type of handler ('email', 'slack', 'webhook', 'log')
            handler: Callable that takes Alert object
        """
        if handler_type not in self.alert_handlers:
            self.alert_handlers[handler_type] = []
        self.alert_handlers[handler_type].append(handler)
        logger.info(f"Handler registered for {handler_type}")
    
    def send_email_alert(self,
                        alert: Alert,
                        smtp_config: Dict[str, str],
                        recipients: List[str]) -> None:
        """
        Send email alert
        
        Args:
            alert: Alert object
            smtp_config: SMTP configuration {'server': '', 'port': 587, 'user': '', 'password': ''}
            recipients: List of email addresses
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config['user']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.upper()}] Model Alert: {alert.metric_name}"
            
            body = f"""
            Model Monitoring Alert
            
            Metric: {alert.metric_name}
            Severity: {alert.severity}
            Current Value: {alert.current_value:.4f}
            Threshold: {alert.threshold:.4f}
            
            Message: {alert.message}
            
            Timestamp: {alert.timestamp}
            Alert ID: {alert.alert_id}
            
            Please review the monitoring dashboard for more details.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_config['server'], smtp_config.get('port', 587))
            server.starttls()
            server.login(smtp_config['user'], smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent to {recipients}")
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def send_slack_alert(self,
                        alert: Alert,
                        webhook_url: str) -> None:
        """
        Send Slack alert
        
        Args:
            alert: Alert object
            webhook_url: Slack webhook URL
        """
        try:
            import requests
            
            color = {
                "info": "#36a64f",
                "warning": "#ff9800",
                "critical": "#f44336"
            }.get(alert.severity, "#999999")
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"Model Alert: {alert.metric_name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.upper(),
                                "short": True
                            },
                            {
                                "title": "Current Value",
                                "value": f"{alert.current_value:.4f}",
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": f"{alert.threshold:.4f}",
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": alert.timestamp,
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            logger.info("Slack alert sent successfully")
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    def get_alert_summary(self,
                         limit: Optional[int] = None) -> Dict[str, Any]:
        """Get summary of alerts"""
        history = self.alerts_history.copy()
        
        if limit:
            history = history[-limit:]
        
        total = len(history)
        by_severity = {}
        for alert in history:
            severity = alert.get("severity", "unknown")
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total_alerts": total,
            "by_severity": by_severity,
            "recent_alerts": history[-10:] if history else [],
            "active_thresholds": len([t for t in self.thresholds.values() if t.enabled])
        }
    
    def acknowledge_alert(self, alert_id: str) -> None:
        """Acknowledge an alert"""
        for alert in self.alerts_history:
            if alert.get("alert_id") == alert_id:
                alert["acknowledged"] = True
                break
        
        try:
            history_file = self.alerts_dir / "alerts_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.alerts_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
    
    def export_alerts(self, export_path: str, format: str = "json") -> None:
        """Export alerts"""
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "json":
                with open(export_path, 'w') as f:
                    json.dump(self.alerts_history, f, indent=2, default=str)
            elif format == "csv":
                import pandas as pd
                df = pd.DataFrame(self.alerts_history)
                df.to_csv(export_path, index=False)
            logger.info(f"Alerts exported to {export_path}")
        except Exception as e:
            logger.error(f"Error exporting alerts: {e}")
