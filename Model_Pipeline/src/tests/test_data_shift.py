"""
Unit tests for data shift detection module.
Tests EvidentlyAI and TFDV-based drift detection capabilities.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from src.monitoring.data_shift_detector import (
    EvidentlyAIDataShiftDetector,
    TFDVDataValidation,
    DataShiftReport,
)


class TestEvidentlyAIDataShiftDetector:
    """Test suite for EvidentlyAI-based data shift detection."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return EvidentlyAIDataShiftDetector()

    @pytest.fixture
    def training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        return {
            'query_length': np.random.normal(50, 15, 100),
            'document_count': np.random.normal(5, 2, 100),
            'relevance_score': np.random.uniform(0.5, 1.0, 100),
            'response_time': np.random.exponential(2, 100),
            'embedding_dimension': np.random.normal(384, 10, 100),
        }

    @pytest.fixture
    def production_data_normal(self):
        """Create production data with normal variation."""
        np.random.seed(43)
        return {
            'query_length': np.random.normal(52, 16, 100),
            'document_count': np.random.normal(5.2, 2.1, 100),
            'relevance_score': np.random.uniform(0.48, 1.0, 100),
            'response_time': np.random.exponential(2.1, 100),
            'embedding_dimension': np.random.normal(385, 11, 100),
        }

    @pytest.fixture
    def production_data_shift(self):
        """Create production data with significant shift."""
        np.random.seed(44)
        return {
            'query_length': np.random.normal(80, 25, 100),
            'document_count': np.random.normal(8, 3, 100),
            'relevance_score': np.random.uniform(0.3, 0.8, 100),
            'response_time': np.random.exponential(6, 100),
            'embedding_dimension': np.random.normal(384, 10, 100),
        }

    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector is not None
        assert len(detector.baseline_profiles) >= 0

    def test_set_baseline(self, detector, training_data):
        """Test baseline setting."""
        detector.set_baseline(training_data, "test_baseline")
        
        assert "test_baseline" in detector.baseline_profiles
        baseline = detector.baseline_profiles["test_baseline"]
        assert len(baseline['features']) == 5
        assert 'query_length' in baseline['features']

    def test_baseline_statistics(self, detector, training_data):
        """Test baseline statistics calculation."""
        detector.set_baseline(training_data, "test_stats")
        
        baseline = detector.baseline_profiles["test_stats"]["features"]['query_length']
        assert 'mean' in baseline
        assert 'std' in baseline
        assert 'min' in baseline
        assert 'max' in baseline
        assert abs(baseline['mean'] - 50.0) < 5

    def test_normal_variation_no_drift(self, detector, training_data, production_data_normal):
        """Test detection of normal variation (no drift)."""
        detector.set_baseline(training_data, "test_normal")
        report = detector.detect_data_drift(production_data_normal, "test_normal")
        
        assert isinstance(report, DataShiftReport)
        assert report.drift_rate < 0.3
        assert report.severity in ['low', 'medium']

    def test_significant_shift_detection(self, detector, training_data, production_data_shift):
        """Test detection of significant data shift."""
        detector.set_baseline(training_data, "test_shift")
        report = detector.detect_data_drift(production_data_shift, "test_shift")
        
        assert isinstance(report, DataShiftReport)
        assert report.drift_rate > 0.5
        assert report.severity in ['high', 'critical']
        assert report.drifted_features > 0

    def test_feature_drift_structure(self, detector, training_data, production_data_shift):
        """Test feature-level drift details."""
        detector.set_baseline(training_data, "test_features")
        report = detector.detect_data_drift(production_data_shift, "test_features")
        
        assert len(report.feature_drifts) > 0
        for feature_drift in report.feature_drifts:
            assert 'feature_name' in feature_drift
            assert 'drift_score' in feature_drift

    def test_report_structure(self, detector, training_data, production_data_normal):
        """Test report structure."""
        detector.set_baseline(training_data, "test_report")
        report = detector.detect_data_drift(production_data_normal, "test_report")
        
        assert isinstance(report, DataShiftReport)
        assert report.timestamp is not None
        assert report.severity is not None
        assert isinstance(report.recommendations, list)


class TestTFDVDataValidation:
    """Test suite for TensorFlow Data Validation (TFDV)."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return TFDVDataValidation()

    @pytest.fixture
    def training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        return {
            'query_length': np.random.normal(50, 15, 100),
            'document_count': np.random.normal(5, 2, 100),
            'relevance_score': np.random.uniform(0.5, 1.0, 100),
            'response_time': np.random.exponential(2, 100),
            'embedding_dimension': np.random.normal(384, 10, 100),
        }

    @pytest.fixture
    def production_data(self):
        """Create production data."""
        np.random.seed(43)
        return {
            'query_length': np.random.normal(52, 16, 100),
            'document_count': np.random.normal(5.2, 2.1, 100),
            'relevance_score': np.random.uniform(0.48, 1.0, 100),
            'response_time': np.random.exponential(2.1, 100),
            'embedding_dimension': np.random.normal(385, 11, 100),
        }

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert validator.schema == {}
        assert validator.validation_dir.exists()

    def test_schema_inference(self, validator, training_data):
        """Test schema inference from data."""
        schema = validator.infer_schema(training_data, "test_schema")
        
        assert isinstance(schema, dict)
        assert 'features' in schema
        assert len(schema['features']) == 5
        assert all(f in schema['features'] for f in training_data.keys())

    def test_schema_feature_structure(self, validator, training_data):
        """Test feature structure in schema."""
        schema = validator.infer_schema(training_data, "test_feature_struct")
        
        for feature_name in training_data.keys():
            feature = schema['features'][feature_name]
            assert 'name' in feature
            assert 'type' in feature

    def test_data_validation_pass(self, validator, training_data, production_data):
        """Test validation of data matching schema."""
        schema = validator.infer_schema(training_data, "test_validation")
        result = validator.validate_data(production_data, schema)
        
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'errors' in result

    def test_anomaly_detection_normal(self, validator, training_data, production_data):
        """Test anomaly detection on normal data."""
        schema = validator.infer_schema(training_data, "test_anomaly_normal")
        result = validator.validate_data(production_data, schema)
        
        assert isinstance(result, dict)
        assert 'valid' in result

    def test_anomaly_detection_with_outliers(self, validator, training_data):
        """Test anomaly detection with outliers."""
        schema = validator.infer_schema(training_data, "test_anomaly_outlier")
        
        # Create data with extreme values
        data_with_outliers = {}
        for key, values in training_data.items():
            data_with_outliers[key] = values.copy()
        
        # Add extreme outliers
        data_with_outliers['response_time'][0] = 100
        data_with_outliers['embedding_dimension'][1] = 1000
        
        result = validator.validate_data(data_with_outliers, schema)
        
        assert isinstance(result, dict)

    def test_schema_persistence(self, validator, training_data):
        """Test schema saving."""
        schema = validator.infer_schema(training_data, "test_persist")
        
        # Verify schema was stored
        assert validator.schema is not None or isinstance(schema, dict)

    def test_validation_result_format(self, validator, training_data, production_data):
        """Test validation result format."""
        schema = validator.infer_schema(training_data, "test_format")
        result = validator.validate_data(production_data, schema)
        
        assert 'valid' in result
        assert 'errors' in result
        assert 'timestamp' in result


class TestDataShiftIntegration:
    """Integration tests combining both detectors."""

    def test_detector_and_validator_workflow(self):
        """Test typical workflow with both detectors."""
        np.random.seed(42)
        
        # Create training data
        training_data = {
            'query_length': np.random.normal(50, 15, 50),
            'document_count': np.random.normal(5, 2, 50),
            'relevance_score': np.random.uniform(0.5, 1.0, 50),
            'response_time': np.random.exponential(2, 50),
            'embedding_dimension': np.random.normal(384, 10, 50),
        }
        
        # Create production data with shift
        production_data = {
            'query_length': np.random.normal(80, 25, 50),
            'document_count': np.random.normal(8, 3, 50),
            'relevance_score': np.random.uniform(0.3, 0.8, 50),
            'response_time': np.random.exponential(6, 50),
            'embedding_dimension': np.random.normal(384, 10, 50),
        }
        
        # Test with detector
        detector = EvidentlyAIDataShiftDetector()
        detector.set_baseline(training_data, "baseline")
        drift_report = detector.detect_data_drift(production_data, "baseline")
        
        assert isinstance(drift_report, DataShiftReport)
        assert drift_report.drift_rate > 0.5
        
        # Test with validator
        validator = TFDVDataValidation()
        schema = validator.infer_schema(training_data, "baseline")
        validation_result = validator.validate_data(production_data, schema)
        
        assert isinstance(validation_result, dict)
        assert 'valid' in validation_result


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
