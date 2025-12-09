import pytest
from scripts.validation.bias_detector import BiasDetector

@pytest.fixture
def sample_transcript():
    return {
        "full_text": "He is a good man. She is a good woman. The guys and girls are here."
    }

def test_term_frequency(sample_transcript):
    detector = BiasDetector()
    freq = detector.term_frequency(sample_transcript["full_text"])
    assert freq["he"] == 1
    assert freq["she"] == 1
    assert freq["man"] == 1
    assert freq["woman"] == 1
    assert freq["guys"] == 1
    assert freq["girls"] == 1

def test_detect_bias(sample_transcript):
    detector = BiasDetector()
    freq = detector.detect_bias(sample_transcript)
    assert sum(freq.values()) == 6

def test_speaker_balance():
    transcript = {
        "segments": [
            {"text": "Hello", "speaker": "Alice"},
            {"text": "Hi", "speaker": "Bob"},
            {"text": "Hey", "speaker": "Alice"}
        ]
    }
    detector = BiasDetector()
    balance = detector.speaker_balance(transcript)
    assert balance["Alice"] == 2
    assert balance["Bob"] == 1
