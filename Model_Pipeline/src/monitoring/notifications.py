"""
Step 5: Stakeholder Notifications for Model Retraining
======================================================

Sends notifications via email and Slack when:
1. Retraining is triggered
2. Model is retrained and validated
3. New model is deployed to production
4. Metrics improve significantly
5. Regressions are detected

Supports:
- Email notifications (SMTP)
- Slack webhooks
- Custom message templates
- Metric summaries and comparisons
"""

import json
import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class NotificationEvent:
    """Notification event"""
    event_type: str  # 'trigger', 'deployed', 'failed', 'metrics_improved'
    timestamp: str
    model_version: str
    trigger_reason: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    comparison: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    recipients: Optional[List[str]] = None


class EmailNotifier:
    """Send email notifications"""
    
    def __init__(self, 
                 smtp_server: str = "smtp.gmail.com",
                 smtp_port: int = 587,
                 sender_email: Optional[str] = None,
                 sender_password: Optional[str] = None):
        """
        Initialize email notifier
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP port
            sender_email: Sender email address
            sender_password: Sender email password (use app password for Gmail)
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        
        logger.info(f"EmailNotifier initialized (server: {smtp_server}:{smtp_port})")
    
    def send_retraining_triggered(self, 
                                 recipients: List[str],
                                 trigger_reason: str,
                                 drift_detected: bool = False,
                                 drift_percentage: float = 0.0) -> bool:
        """Send notification that retraining was triggered"""
        try:
            subject = "🔄 MLOps Alert: Model Retraining Triggered"
            
            body = f"""
            <html>
              <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #FF9800;">Model Retraining Triggered</h2>
                
                <p><strong>Timestamp:</strong> {datetime.now().isoformat()}</p>
                
                <h3>Trigger Reason:</h3>
                <p style="background-color: #FFF3E0; padding: 10px; border-radius: 5px;">
                  {trigger_reason}
                </p>
                
                {f"<p><strong>Data Drift Detected:</strong> {drift_percentage:.1f}%</p>" if drift_detected else ""}
                
                <h3>Next Steps:</h3>
                <ul>
                  <li>New data is being pulled from GCS</li>
                  <li>Model retraining in progress</li>
                  <li>Validation checks running</li>
                  <li>You will receive another notification when deployment decision is made</li>
                </ul>
                
                <hr/>
                <p style="color: #666; font-size: 12px;">
                  This is an automated notification from MLOps monitoring system.
                </p>
              </body>
            </html>
            """
            
            return self._send_email(recipients, subject, body)
        
        except Exception as e:
            logger.error(f"Failed to send retraining triggered email: {e}")
            return False
    
    def send_model_deployed(self,
                           recipients: List[str],
                           old_version: str,
                           new_version: str,
                           improvements: Dict[str, float]) -> bool:
        """Send notification that new model was deployed"""
        try:
            subject = "✅ Model Successfully Deployed to Production"
            
            improvements_html = "".join([
                f'<tr><td style="padding: 8px;">{metric}</td><td style="padding: 8px; color: green;">+{pct:.1f}%</td></tr>'
                for metric, pct in improvements.items()
            ])
            
            avg_improvement = sum(improvements.values()) / len(improvements)
            
            body = f"""
            <html>
              <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #4CAF50;">✅ Model Successfully Deployed</h2>
                
                <p><strong>Timestamp:</strong> {datetime.now().isoformat()}</p>
                
                <h3>Version Information:</h3>
                <table style="border-collapse: collapse;">
                  <tr style="background-color: #E8F5E9;">
                    <td style="padding: 8px;"><strong>Previous Version:</strong></td>
                    <td style="padding: 8px;">{old_version}</td>
                  </tr>
                  <tr>
                    <td style="padding: 8px;"><strong>New Version:</strong></td>
                    <td style="padding: 8px;">{new_version}</td>
                  </tr>
                </table>
                
                <h3>Performance Improvements:</h3>
                <table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
                  <tr style="background-color: #E3F2FD;">
                    <th style="padding: 8px; text-align: left;">Metric</th>
                    <th style="padding: 8px; text-align: left;">Improvement</th>
                  </tr>
                  {improvements_html}
                  <tr style="background-color: #E8F5E9; font-weight: bold;">
                    <td style="padding: 8px;">Average Improvement</td>
                    <td style="padding: 8px; color: green;">+{avg_improvement:.1f}%</td>
                  </tr>
                </table>
                
                <h3>Impact:</h3>
                <ul>
                  <li>Better search accuracy and relevance</li>
                  <li>Faster response times</li>
                  <li>Improved user satisfaction</li>
                  <li>All metrics above production thresholds</li>
                </ul>
                
                <h3>Rollback Plan:</h3>
                <p>Previous model {old_version} has been archived and can be restored if needed.</p>
                
                <hr/>
                <p style="color: #666; font-size: 12px;">
                  This is an automated notification from MLOps monitoring system.
                </p>
              </body>
            </html>
            """
            
            return self._send_email(recipients, subject, body)
        
        except Exception as e:
            logger.error(f"Failed to send deployment email: {e}")
            return False
    
    def send_deployment_held(self,
                            recipients: List[str],
                            reason: str,
                            metrics: Dict[str, float]) -> bool:
        """Send notification that deployment was held"""
        try:
            subject = "⏸️ Model Deployment Held - Review Needed"
            
            body = f"""
            <html>
              <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #FF9800;">⏸️ Deployment Held</h2>
                
                <p><strong>Timestamp:</strong> {datetime.now().isoformat()}</p>
                
                <h3>Reason for Hold:</h3>
                <p style="background-color: #FFF3E0; padding: 10px; border-radius: 5px;">
                  {reason}
                </p>
                
                <h3>Current Metrics:</h3>
                <ul>
                  {''.join([f'<li><strong>{k}:</strong> {v:.3f}</li>' for k, v in metrics.items()])}
                </ul>
                
                <h3>Recommended Actions:</h3>
                <ol>
                  <li>Review the retraining results</li>
                  <li>Check if additional data is needed</li>
                  <li>Investigate potential data quality issues</li>
                  <li>Consider manual model adjustments</li>
                  <li>Re-run retraining with different parameters</li>
                </ol>
                
                <hr/>
                <p style="color: #666; font-size: 12px;">
                  This is an automated notification from MLOps monitoring system.
                </p>
              </body>
            </html>
            """
            
            return self._send_email(recipients, subject, body)
        
        except Exception as e:
            logger.error(f"Failed to send hold email: {e}")
            return False
    
    def send_retraining_failed(self,
                              recipients: List[str],
                              error_message: str) -> bool:
        """Send notification that retraining failed"""
        try:
            subject = "❌ Model Retraining Failed"
            
            body = f"""
            <html>
              <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #F44336;">❌ Retraining Failed</h2>
                
                <p><strong>Timestamp:</strong> {datetime.now().isoformat()}</p>
                
                <h3>Error Details:</h3>
                <pre style="background-color: #FFEBEE; padding: 10px; border-radius: 5px; overflow-x: auto;">
{error_message}
                </pre>
                
                <h3>Immediate Actions Required:</h3>
                <ul>
                  <li>Check data quality and availability</li>
                  <li>Review error logs in MLOps dashboard</li>
                  <li>Verify GCS bucket connectivity</li>
                  <li>Contact MLOps team if issue persists</li>
                </ul>
                
                <p><strong>Production Status:</strong> Current model v1.0.0 remains in production</p>
                
                <hr/>
                <p style="color: #666; font-size: 12px;">
                  This is an automated notification from MLOps monitoring system.
                </p>
              </body>
            </html>
            """
            
            return self._send_email(recipients, subject, body)
        
        except Exception as e:
            logger.error(f"Failed to send failure email: {e}")
            return False
    
    def _send_email(self, recipients: List[str], subject: str, body: str) -> bool:
        """Internal method to send email"""
        try:
            if not self.sender_email or not self.sender_password:
                logger.warning("Email credentials not configured - skipping email notification")
                return True  # Don't fail the system if email is not configured
            
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.sender_email
            msg["To"] = ", ".join(recipients)
            
            # Attach HTML body
            msg.attach(MIMEText(body, "html"))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipients, msg.as_string())
            
            logger.info(f"Email sent to {recipients}: {subject}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


class SlackNotifier:
    """Send Slack notifications"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Slack notifier
        
        Args:
            webhook_url: Slack webhook URL for posting messages
        """
        self.webhook_url = webhook_url
        logger.info(f"SlackNotifier initialized (webhook configured: {bool(webhook_url)})")
    
    def send_retraining_triggered(self,
                                 trigger_reason: str,
                                 drift_percentage: float = 0.0) -> bool:
        """Send Slack notification that retraining was triggered"""
        try:
            if not self.webhook_url:
                logger.warning("Slack webhook not configured")
                return True
            
            message = {
                "text": "🔄 Model Retraining Triggered",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "🔄 Model Retraining Triggered"}
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Timestamp:*\n{datetime.now().isoformat()}"},
                            {"type": "mrkdwn", "text": f"*Status:*\n🟡 In Progress"}
                        ]
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Trigger Reason:*\n{trigger_reason}"}
                    }
                ]
            }
            
            if drift_percentage > 0:
                message["blocks"].append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Data Drift Detected:* {drift_percentage:.1f}%"}
                })
            
            return self._send_slack(message)
        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def send_model_deployed(self,
                           old_version: str,
                           new_version: str,
                           improvements: Dict[str, float]) -> bool:
        """Send Slack notification that new model was deployed"""
        try:
            if not self.webhook_url:
                logger.warning("Slack webhook not configured")
                return True
            
            avg_improvement = sum(improvements.values()) / len(improvements)
            
            improvements_text = "\n".join([
                f"• {metric}: +{pct:.1f}%" for metric, pct in improvements.items()
            ])
            
            message = {
                "text": "✅ Model Successfully Deployed",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "✅ Model Successfully Deployed"}
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Previous Version:*\n{old_version}"},
                            {"type": "mrkdwn", "text": f"*New Version:*\n{new_version}"}
                        ]
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Performance Improvements:*\n{improvements_text}"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Average Improvement:* +{avg_improvement:.1f}%"}
                    }
                ]
            }
            
            return self._send_slack(message)
        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def send_deployment_held(self, reason: str) -> bool:
        """Send Slack notification that deployment was held"""
        try:
            if not self.webhook_url:
                logger.warning("Slack webhook not configured")
                return True
            
            message = {
                "text": "⏸️ Deployment Held",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "⏸️ Deployment Held"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Reason:*\n{reason}"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "Review needed before proceeding"}
                    }
                ]
            }
            
            return self._send_slack(message)
        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def send_retraining_failed(self, error_message: str) -> bool:
        """Send Slack notification that retraining failed"""
        try:
            if not self.webhook_url:
                logger.warning("Slack webhook not configured")
                return True
            
            message = {
                "text": "❌ Retraining Failed",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "❌ Retraining Failed"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Error:*\n```{error_message[:200]}```"}
                    }
                ]
            }
            
            return self._send_slack(message)
        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _send_slack(self, message: Dict[str, Any]) -> bool:
        """Internal method to send Slack message"""
        try:
            response = requests.post(self.webhook_url, json=message)
            if response.status_code == 200:
                logger.info("Slack message sent successfully")
                return True
            else:
                logger.error(f"Slack API error: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False


class NotificationManager:
    """Manage all notifications for retraining pipeline"""
    
    def __init__(self,
                 email_recipients: List[str],
                 smtp_server: str = "smtp.gmail.com",
                 smtp_port: int = 587,
                 sender_email: Optional[str] = None,
                 sender_password: Optional[str] = None,
                 slack_webhook_url: Optional[str] = None):
        """
        Initialize notification manager
        
        Args:
            email_recipients: List of email addresses to notify
            smtp_server: SMTP server
            smtp_port: SMTP port
            sender_email: Sender email address
            sender_password: Sender email password
            slack_webhook_url: Slack webhook URL
        """
        self.email_recipients = email_recipients
        self.email_notifier = EmailNotifier(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            sender_email=sender_email,
            sender_password=sender_password
        )
        self.slack_notifier = SlackNotifier(webhook_url=slack_webhook_url)
        self.notification_history = []
        
        logger.info(f"NotificationManager initialized with {len(email_recipients)} recipients")
    
    def notify_retraining_triggered(self,
                                   trigger_reason: str,
                                   drift_detected: bool = False,
                                   drift_percentage: float = 0.0) -> bool:
        """Notify stakeholders that retraining was triggered"""
        logger.info(f"Sending retraining triggered notification")
        
        # Send email
        email_sent = self.email_notifier.send_retraining_triggered(
            recipients=self.email_recipients,
            trigger_reason=trigger_reason,
            drift_detected=drift_detected,
            drift_percentage=drift_percentage
        )
        
        # Send Slack
        slack_sent = self.slack_notifier.send_retraining_triggered(
            trigger_reason=trigger_reason,
            drift_percentage=drift_percentage
        )
        
        # Log notification
        self.notification_history.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'retraining_triggered',
            'email_sent': email_sent,
            'slack_sent': slack_sent
        })
        
        return email_sent or slack_sent
    
    def notify_model_deployed(self,
                             old_version: str,
                             new_version: str,
                             improvements: Dict[str, float]) -> bool:
        """Notify stakeholders that new model was deployed"""
        logger.info(f"Sending deployment notification: {old_version} → {new_version}")
        
        # Send email
        email_sent = self.email_notifier.send_model_deployed(
            recipients=self.email_recipients,
            old_version=old_version,
            new_version=new_version,
            improvements=improvements
        )
        
        # Send Slack
        slack_sent = self.slack_notifier.send_model_deployed(
            old_version=old_version,
            new_version=new_version,
            improvements=improvements
        )
        
        # Log notification
        self.notification_history.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'model_deployed',
            'old_version': old_version,
            'new_version': new_version,
            'email_sent': email_sent,
            'slack_sent': slack_sent
        })
        
        return email_sent or slack_sent
    
    def notify_deployment_held(self,
                              reason: str,
                              metrics: Dict[str, float]) -> bool:
        """Notify stakeholders that deployment was held"""
        logger.info(f"Sending deployment held notification")
        
        # Send email
        email_sent = self.email_notifier.send_deployment_held(
            recipients=self.email_recipients,
            reason=reason,
            metrics=metrics
        )
        
        # Send Slack
        slack_sent = self.slack_notifier.send_deployment_held(reason=reason)
        
        # Log notification
        self.notification_history.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'deployment_held',
            'reason': reason,
            'email_sent': email_sent,
            'slack_sent': slack_sent
        })
        
        return email_sent or slack_sent
    
    def notify_retraining_failed(self, error_message: str) -> bool:
        """Notify stakeholders that retraining failed"""
        logger.info(f"Sending retraining failed notification")
        
        # Send email
        email_sent = self.email_notifier.send_retraining_failed(
            recipients=self.email_recipients,
            error_message=error_message
        )
        
        # Send Slack
        slack_sent = self.slack_notifier.send_retraining_failed(error_message=error_message)
        
        # Log notification
        self.notification_history.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'retraining_failed',
            'error': error_message[:100],
            'email_sent': email_sent,
            'slack_sent': slack_sent
        })
        
        return email_sent or slack_sent
    
    def get_notification_history(self) -> List[Dict[str, Any]]:
        """Get notification history"""
        return self.notification_history
    
    def export_notification_history(self, filepath: str) -> bool:
        """Export notification history to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.notification_history, f, indent=2)
            logger.info(f"Notification history exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export notification history: {e}")
            return False
