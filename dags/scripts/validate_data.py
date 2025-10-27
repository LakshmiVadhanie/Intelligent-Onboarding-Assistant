import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2] / "data"

def validate_json_structure(file_path, required_keys):
    """Validate JSON file structure and contents."""
    if not file_path.exists():
        print(f"Missing file: {file_path}")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        print(f"{file_path.name}: Empty or invalid list format")
        return False

    for i, item in enumerate(data[:5]):  # check first few
        if not all(k in item for k in required_keys):
            print(f"{file_path.name}: Missing keys in item {i}")
            return False
        for key in required_keys:
            if not item[key] or not str(item[key]).strip():
                print(f"{file_path.name}: Empty value for key '{key}' in item {i}")
                return False

    print(f"{file_path.name}: Structure and content look valid ({len(data)} records).")
    return True


def run_validation():
    handbook_path = BASE_DIR / "handbook_paragraphs.json"
    transcript_path = BASE_DIR / "meeting_transcripts" / "all_transcripts.json"

    checks = [
        validate_json_structure(handbook_path, ["title", "paragraph"]),
        validate_json_structure(transcript_path, ["title", "video_id", "url", "transcript"]),
    ]

    if all(checks):
        print("All validation checks passed successfully!")
        return True
    else:
        print("Validation failed — inspect logs for details.")
        return False


if __name__ == "__main__":
    success = run_validation()
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
