from pathlib import Path
import os, mimetypes
from dotenv import load_dotenv
from google.cloud import storage

# Load .env from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

BUCKET = os.getenv("GCS_BUCKET", "mlops-data-bucket-oa")

def upload_folder(local_dir: str, gcs_prefix: str) -> int:
    base = Path(local_dir)
    if not base.exists():
        raise FileNotFoundError(f"Not found: {base}")
    client = storage.Client()  # uses GOOGLE_APPLICATION_CREDENTIALS from .env
    bucket = client.bucket(BUCKET)
    n = 0
    for p in base.rglob("*"):
        if p.is_file():
            rel = p.relative_to(base).as_posix()
            obj = f"{gcs_prefix.rstrip('/')}/{rel}"
            blob = bucket.blob(obj)
            ctype, _ = mimetypes.guess_type(p.as_posix())
            if ctype: blob.content_type = ctype
            blob.cache_control = "no-cache"
            blob.upload_from_filename(p.as_posix())
            n += 1
    print(f"✅ Uploaded {n} files from {base} → gs://{BUCKET}/{gcs_prefix}/")
    return n

if __name__ == "__main__":
    upload_folder("data-pipeline/data/raw", "raw")
    upload_folder("data-pipeline/data/processed", "processed")
