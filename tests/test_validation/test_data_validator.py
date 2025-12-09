import pytest
from datetime import datetime
from scripts.validation.data_validator import DataValidator

@pytest.fixture
def sample_data():
    return {
        "title": "Test Meeting",
        "video_id": "test123",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.0,
                "text": "First segment"
            },
            {
                "id": 1,
                "start": 5.0,
                "end": 10.0,
                "text": "Second segment"
            }
        ]
    }

@pytest.fixture
def validator():
    return DataValidator()

def test_schema_generation(validator, sample_data):
    schema = validator.generate_schema(sample_data)
    assert "required" in schema
    assert "properties" in schema
    assert "segments" in schema["properties"]

def test_validate_schema(validator, sample_data):
    validator.schema = validator.generate_schema(sample_data)
    errors = validator.validate_schema(sample_data)
    assert len(errors) == 0

def test_validate_schema_with_errors(validator):
    invalid_data = {
        "title": "Test",
        # missing video_id
        "segments": [
            {
                "id": 0,
                "start": 5.0,
                "end": 0.0,  # invalid time range
                "text": "Test"
            }
        ]
    }
    validator.schema = validator.generate_schema(invalid_data)
    errors = validator.validate_schema(invalid_data)
    assert len(errors) > 0

def test_statistics_generation(validator, sample_data):
    stats = validator.generate_statistics(sample_data)
    assert stats["total_segments"] == 2
    assert stats["total_words"] == 4
    assert stats["avg_segment_duration"] == 5.0
    assert stats["missing_values"] == 0

def test_anomaly_detection(validator, sample_data):
    # Generate initial stats
    stats1 = validator.generate_statistics(sample_data)
    
    # Create data with anomaly
    anomaly_data = sample_data.copy()
    anomaly_data["segments"] = anomaly_data["segments"][:1]  # Remove 50% of segments
    
    stats2 = validator.generate_statistics(anomaly_data)
    anomalies = validator.detect_anomalies(stats2)
    
    assert len(anomalies) > 0
    assert any("total_segments" in anomaly for anomaly in anomalies)