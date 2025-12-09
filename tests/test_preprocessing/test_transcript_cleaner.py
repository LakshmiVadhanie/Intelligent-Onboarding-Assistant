import pytest
import json
from pathlib import Path
from scripts.preprocessing.transcript_cleaner import TranscriptCleaner

@pytest.fixture
def sample_transcript():
    return {
        "title": "Test Meeting",
        "video_id": "test123",
        "url": "https://example.com",
        "transcript": "Um... let me see. Yeah, you know, like, the thing is...",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.0,
                "text": "Um... let me see.",
                "tokens": [1, 2, 3]
            },
            {
                "id": 1,
                "start": 5.0,
                "end": 10.0,
                "text": "Yeah, you know, like, the thing is...",
                "tokens": [4, 5, 6]
            }
        ]
    }

@pytest.fixture
def cleaner():
    return TranscriptCleaner()

def test_clean_text():
    cleaner = TranscriptCleaner()
    text = "Um... yeah, you know, like, this is a test."
    cleaned = cleaner.clean_text(text)
    assert "um" not in cleaned.lower()
    assert "you know" not in cleaned.lower()
    assert "like" not in cleaned.lower()
    assert "this is a test" in cleaned.lower()

def test_merge_segments():
    cleaner = TranscriptCleaner()
    segments = [
        {"text": "First part", "start": 0.0, "end": 5.0},
        {"text": "Second part", "start": 5.0, "end": 10.0}
    ]
    merged = cleaner.merge_segments(segments)
    assert len(merged) < len(segments)
    assert "First part" in merged[0]["text"]
    assert "Second part" in merged[0]["text"]

def test_handle_missing_segments():
    cleaner = TranscriptCleaner()
    transcript = {
        "title": "Test",
        "video_id": "123",
        "transcript": "Test"
    }
    with pytest.raises(KeyError):
        cleaner.process_transcript(transcript)

def test_empty_transcript():
    cleaner = TranscriptCleaner()
    transcript = {
        "title": "Test",
        "video_id": "123",
        "segments": []
    }
    processed = cleaner.process_transcript(transcript)
    assert processed["segment_count"] == 0
    assert processed["word_count"] == 0

def test_full_pipeline(sample_transcript, tmp_path):
    cleaner = TranscriptCleaner()
    input_file = tmp_path / "test_input.json"
    output_file = tmp_path / "cleaned_test_input.json"
    
    with open(input_file, "w") as f:
        json.dump(sample_transcript, f)
    
    cleaner.process_file(str(input_file), str(output_file))
    
    assert output_file.exists()
    with open(output_file) as f:
        cleaned = json.load(f)
    
    assert "cleaned_at" in cleaned
    assert cleaned["segment_count"] < len(sample_transcript["segments"])
    assert "um" not in cleaned["full_text"].lower()