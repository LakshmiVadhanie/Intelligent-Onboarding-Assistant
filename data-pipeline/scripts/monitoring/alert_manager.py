import smtplib
import requests
from email.mime.text import MIMEText
from typing import List, Optional
import os
import json
from pathlib import Path
import logging
from .logging_setup import setup_logger

logger = setup_logger('monitoring')

class AlertManager:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize AlertManager with configuration"""
        self.config = self._load_config(config_path) if config_path else {}
        
    def _load_config(self, config_path: str) -> dict:
        """Load alerting configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def send_email_alert(self, subject: str, message: str):
        """Send email alert"""
        try:
            if 'email' not in self.config:
                logger.warning("Email configuration not found")
                return
                
            email_config = self.config['email']
            
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = email_config['from']
            msg['To'] = email_config['to']
            
            with smtplib.SMTP_SSL(email_config['smtp_server']) as server:
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
                
            logger.info(f"Email alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
    
    def send_slack_alert(self, message: str):
        """Send Slack alert"""
        try:
            if 'slack' not in self.config:
                logger.warning("Slack configuration not found")
                return
                
            slack_config = self.config['slack']
            webhook_url = slack_config['webhook_url']
            
            response = requests.post(
                webhook_url,
                json={"text": message}
            )
            
            if response.status_code == 200:
                logger.info("Slack alert sent successfully")
            else:
                logger.error(f"Failed to send Slack alert: {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
    
    def handle_anomalies(self, anomalies: List[str]):
        """Handle detected anomalies"""
        if not anomalies:
            return
            
        # Prepare alert message
        message = "Data Pipeline Anomalies Detected:\n\n"
        message += "\n".join(f"- {anomaly}" for anomaly in anomalies)
        
        # Send alerts through configured channels
        self.send_email_alert("Data Pipeline Anomalies Detected", message)
        self.send_slack_alert(message)
        
        # Log the anomalies
        logger.warning(f"Anomalies detected:\n{message}")
    
    def handle_pipeline_error(self, error: str, step: str):
        """Handle pipeline execution error"""
        message = f"Pipeline Error in step '{step}':\n{error}"
        
        # Send alerts
        self.send_email_alert(f"Pipeline Error - {step}", message)
        self.send_slack_alert(message)
        
        # Log the error
        logger.error(f"Pipeline error in {step}: {error}")
    
    def save_alert_history(self, alert_dir: str = "data-pipeline/monitoring/alerts"):
        """Save alert history to file"""
        alert_dir = Path(alert_dir)
        alert_dir.mkdir(parents=True, exist_ok=True)
        
        history_file = alert_dir / "alert_history.json"
        
        # Create history entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "alerts": self._get_recent_alerts()
        }
        
        # Load existing history
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
            
        # Append new entry
        history.append(entry)
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _get_recent_alerts(self) -> List[dict]:
        """Get recent alerts from log files"""
        # Implementation depends on your logging setup
        # This is a placeholder that should be customized
        return []