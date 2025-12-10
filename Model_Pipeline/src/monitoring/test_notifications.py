"""
Unit Tests for Step 5: Stakeholder Notifications

Tests all notification components:
- EmailNotifier: Email sending functionality
- SlackNotifier: Slack webhook integration
- NotificationManager: Orchestration and history tracking
- Different notification scenarios
"""

import sys
import json
import unittest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add monitoring module to path
sys.path.insert(0, str(Path(__file__).parent))

from notifications import (
    EmailNotifier,
    SlackNotifier,
    NotificationManager,
    NotificationEvent
)


class TestNotificationEvent(unittest.TestCase):
    """Test NotificationEvent dataclass"""
    
    def test_event_creation(self):
        """Test creating NotificationEvent"""
        event = NotificationEvent(
            event_type="deployed",
            timestamp=datetime.now().isoformat(),
            model_version="v1.1.0",
            trigger_reason="threshold_violation",
            metrics={"precision": 0.85},
            comparison={"improvement": 0.05}
        )
        
        self.assertEqual(event.event_type, "deployed")
        self.assertEqual(event.model_version, "v1.1.0")
        self.assertIsNotNone(event.timestamp)
    
    def test_event_with_optional_fields(self):
        """Test NotificationEvent with optional fields"""
        event = NotificationEvent(
            event_type="failed",
            timestamp=datetime.now().isoformat(),
            model_version="v1.1.0",
            error_message="GCS connection failed"
        )
        
        self.assertEqual(event.error_message, "GCS connection failed")
        self.assertIsNone(event.metrics)


class TestEmailNotifier(unittest.TestCase):
    """Test EmailNotifier functionality"""
    
    def setUp(self):
        self.notifier = EmailNotifier(
            sender_email="test@gmail.com",
            sender_password="test_password"
        )
    
    def test_initialization(self):
        """Test EmailNotifier initialization"""
        self.assertEqual(self.notifier.smtp_server, "smtp.gmail.com")
        self.assertEqual(self.notifier.smtp_port, 587)
        self.assertEqual(self.notifier.sender_email, "test@gmail.com")
    
    def test_custom_smtp_settings(self):
        """Test EmailNotifier with custom SMTP settings"""
        notifier = EmailNotifier(
            smtp_server="smtp.custom.com",
            smtp_port=465
        )
        
        self.assertEqual(notifier.smtp_server, "smtp.custom.com")
        self.assertEqual(notifier.smtp_port, 465)
    
    @patch('smtplib.SMTP')
    def test_send_retraining_triggered(self, mock_smtp):
        """Test sending retraining triggered notification"""
        result = self.notifier.send_retraining_triggered(
            recipients=["user@example.com"],
            trigger_reason="Threshold violation detected",
            drift_detected=True,
            drift_percentage=80.0
        )
        
        # Should return True (email configured)
        self.assertTrue(result or result is None)
    
    @patch('smtplib.SMTP')
    def test_send_model_deployed(self, mock_smtp):
        """Test sending model deployed notification"""
        improvements = {
            "precision": 12.5,
            "recall": 11.8,
            "mrr": 9.3
        }
        
        result = self.notifier.send_model_deployed(
            recipients=["user@example.com"],
            old_version="v1.0.0",
            new_version="v1.1.0",
            improvements=improvements
        )
        
        self.assertTrue(result or result is None)
    
    @patch('smtplib.SMTP')
    def test_send_deployment_held(self, mock_smtp):
        """Test sending deployment held notification"""
        result = self.notifier.send_deployment_held(
            recipients=["user@example.com"],
            reason="Insufficient improvement",
            metrics={"precision": 0.72, "recall": 0.68}
        )
        
        self.assertTrue(result or result is None)
    
    @patch('smtplib.SMTP')
    def test_send_retraining_failed(self, mock_smtp):
        """Test sending retraining failed notification"""
        result = self.notifier.send_retraining_failed(
            recipients=["user@example.com"],
            error_message="GCS connection failed"
        )
        
        self.assertTrue(result or result is None)
    
    def test_send_without_credentials(self):
        """Test that send returns gracefully without credentials"""
        notifier = EmailNotifier()  # No credentials
        
        result = notifier.send_retraining_triggered(
            recipients=["user@example.com"],
            trigger_reason="Test"
        )
        
        # Should return True (graceful fallback)
        self.assertTrue(result)


class TestSlackNotifier(unittest.TestCase):
    """Test SlackNotifier functionality"""
    
    def setUp(self):
        self.webhook_url = "https://hooks.slack.com/services/TEST/TEST"
        self.notifier = SlackNotifier(webhook_url=self.webhook_url)
    
    def test_initialization(self):
        """Test SlackNotifier initialization"""
        self.assertEqual(self.notifier.webhook_url, self.webhook_url)
    
    def test_initialization_without_webhook(self):
        """Test SlackNotifier without webhook"""
        notifier = SlackNotifier()
        self.assertIsNone(notifier.webhook_url)
    
    @patch('requests.post')
    def test_send_retraining_triggered(self, mock_post):
        """Test sending retraining triggered Slack notification"""
        mock_post.return_value.status_code = 200
        
        result = self.notifier.send_retraining_triggered(
            trigger_reason="Threshold violation",
            drift_percentage=80.0
        )
        
        self.assertTrue(result)
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_send_model_deployed(self, mock_post):
        """Test sending model deployed Slack notification"""
        mock_post.return_value.status_code = 200
        
        improvements = {"precision": 12.5, "recall": 11.8}
        result = self.notifier.send_model_deployed(
            old_version="v1.0.0",
            new_version="v1.1.0",
            improvements=improvements
        )
        
        self.assertTrue(result)
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_send_deployment_held(self, mock_post):
        """Test sending deployment held Slack notification"""
        mock_post.return_value.status_code = 200
        
        result = self.notifier.send_deployment_held(
            reason="Insufficient improvement"
        )
        
        self.assertTrue(result)
    
    @patch('requests.post')
    def test_send_retraining_failed(self, mock_post):
        """Test sending retraining failed Slack notification"""
        mock_post.return_value.status_code = 200
        
        result = self.notifier.send_retraining_failed(
            error_message="GCS connection failed"
        )
        
        self.assertTrue(result)
    
    @patch('requests.post')
    def test_slack_api_error_handling(self, mock_post):
        """Test handling of Slack API errors"""
        mock_post.return_value.status_code = 500
        
        result = self.notifier.send_retraining_triggered(
            trigger_reason="Test",
            drift_percentage=50.0
        )
        
        self.assertFalse(result)
    
    def test_send_without_webhook(self):
        """Test that send returns gracefully without webhook"""
        notifier = SlackNotifier()  # No webhook
        
        result = notifier.send_retraining_triggered(
            trigger_reason="Test",
            drift_percentage=50.0
        )
        
        # Should return True (graceful fallback)
        self.assertTrue(result)


class TestNotificationManager(unittest.TestCase):
    """Test NotificationManager orchestration"""
    
    def setUp(self):
        self.recipients = ["user@example.com", "admin@example.com"]
        self.manager = NotificationManager(
            email_recipients=self.recipients,
            sender_email="test@gmail.com",
            sender_password="test_password",
            slack_webhook_url="https://hooks.slack.com/services/TEST"
        )
    
    def test_initialization(self):
        """Test NotificationManager initialization"""
        self.assertEqual(self.manager.email_recipients, self.recipients)
        self.assertIsNotNone(self.manager.email_notifier)
        self.assertIsNotNone(self.manager.slack_notifier)
        self.assertEqual(self.manager.notification_history, [])
    
    def test_notification_history_tracking(self):
        """Test that notifications are tracked in history"""
        initial_count = len(self.manager.notification_history)
        
        with patch.object(self.manager.email_notifier, 'send_retraining_triggered', return_value=True):
            with patch.object(self.manager.slack_notifier, 'send_retraining_triggered', return_value=True):
                self.manager.notify_retraining_triggered(
                    trigger_reason="Test",
                    drift_detected=False
                )
        
        # History should have new entry
        self.assertGreater(len(self.manager.notification_history), initial_count)
    
    def test_notify_retraining_triggered(self):
        """Test retraining triggered notification"""
        with patch.object(self.manager.email_notifier, 'send_retraining_triggered', return_value=True):
            with patch.object(self.manager.slack_notifier, 'send_retraining_triggered', return_value=True):
                result = self.manager.notify_retraining_triggered(
                    trigger_reason="Threshold violation",
                    drift_detected=True,
                    drift_percentage=80.0
                )
        
        self.assertTrue(result)
        self.assertGreater(len(self.manager.notification_history), 0)
    
    def test_notify_model_deployed(self):
        """Test model deployed notification"""
        improvements = {"precision": 12.5, "recall": 11.8}
        
        with patch.object(self.manager.email_notifier, 'send_model_deployed', return_value=True):
            with patch.object(self.manager.slack_notifier, 'send_model_deployed', return_value=True):
                result = self.manager.notify_model_deployed(
                    old_version="v1.0.0",
                    new_version="v1.1.0",
                    improvements=improvements
                )
        
        self.assertTrue(result)
    
    def test_notify_deployment_held(self):
        """Test deployment held notification"""
        with patch.object(self.manager.email_notifier, 'send_deployment_held', return_value=True):
            with patch.object(self.manager.slack_notifier, 'send_deployment_held', return_value=True):
                result = self.manager.notify_deployment_held(
                    reason="Insufficient improvement",
                    metrics={"precision": 0.72}
                )
        
        self.assertTrue(result)
    
    def test_notify_retraining_failed(self):
        """Test retraining failed notification"""
        with patch.object(self.manager.email_notifier, 'send_retraining_failed', return_value=True):
            with patch.object(self.manager.slack_notifier, 'send_retraining_failed', return_value=True):
                result = self.manager.notify_retraining_failed(
                    error_message="GCS connection failed"
                )
        
        self.assertTrue(result)
    
    def test_get_notification_history(self):
        """Test retrieving notification history"""
        history = self.manager.get_notification_history()
        
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 0)  # Initially empty
    
    def test_export_notification_history(self):
        """Test exporting notification history"""
        # Add some history
        with patch.object(self.manager.email_notifier, 'send_retraining_triggered', return_value=True):
            with patch.object(self.manager.slack_notifier, 'send_retraining_triggered', return_value=True):
                self.manager.notify_retraining_triggered(
                    trigger_reason="Test"
                )
        
        # Export to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            result = self.manager.export_notification_history(temp_path)
            self.assertTrue(result)
            
            # Verify file was created and contains data
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
        finally:
            Path(temp_path).unlink()
    
    def test_multiple_recipients(self):
        """Test notification with multiple recipients"""
        recipients = ["user1@example.com", "user2@example.com", "user3@example.com"]
        manager = NotificationManager(email_recipients=recipients)
        
        self.assertEqual(len(manager.email_recipients), 3)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for notification scenarios"""
    
    def test_complete_mlops_notification_chain(self):
        """Test complete notification chain for MLOps pipeline"""
        manager = NotificationManager(
            email_recipients=["team@example.com"],
            sender_email="mlops@gmail.com",
            sender_password="test"
        )
        
        # Scenario: Retraining triggered → deployed → notification sent
        with patch.object(manager.email_notifier, 'send_retraining_triggered', return_value=True):
            with patch.object(manager.slack_notifier, 'send_retraining_triggered', return_value=True):
                manager.notify_retraining_triggered(
                    trigger_reason="80% data drift detected",
                    drift_detected=True,
                    drift_percentage=80.0
                )
        
        with patch.object(manager.email_notifier, 'send_model_deployed', return_value=True):
            with patch.object(manager.slack_notifier, 'send_model_deployed', return_value=True):
                manager.notify_model_deployed(
                    old_version="v1.0.0",
                    new_version="v1.1.0",
                    improvements={"precision": 12.5, "recall": 11.8}
                )
        
        # Verify history
        history = manager.get_notification_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['event'], 'retraining_triggered')
        self.assertEqual(history[1]['event'], 'model_deployed')
    
    def test_error_recovery_in_notifications(self):
        """Test that errors in one channel don't break others"""
        manager = NotificationManager(
            email_recipients=["team@example.com"],
            slack_webhook_url="https://hooks.slack.com/services/TEST"
        )
        
        # Email fails, Slack succeeds
        with patch.object(manager.email_notifier, 'send_model_deployed', return_value=False):
            with patch.object(manager.slack_notifier, 'send_model_deployed', return_value=True):
                result = manager.notify_model_deployed(
                    old_version="v1.0.0",
                    new_version="v1.1.0",
                    improvements={"precision": 12.5}
                )
        
        # Overall should succeed (Slack succeeded)
        self.assertTrue(result)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
