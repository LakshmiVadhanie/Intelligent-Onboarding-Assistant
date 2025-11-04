"""
Upload Debiased Data to Google Cloud Storage (GCS)
==================================================

Uploads all final debiased JSON files (handbook + transcripts)
from the local data directory to a specified GCS bucket.

Requires:
    pip install google-cloud-storage
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account.json"
"""

import os
from pathlib import Path
from google.cloud import storage
import logging

# ---------------- Logging Setup ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gcs_upload")

# ---------------- Configuration ---------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "debiased_data"

# 🔧 CHANGE THIS to your actual GCS bucket name
BUCKET_NAME = "mlops-data-oa"
GCS_FOLDER = "onboarding_ai/debiased_data"  # optional folder path inside the bucket


# ---------------- Upload Function ---------------- #
def upload_file_to_gcs(bucket: storage.Bucket, source_file: Path, destination_blob: str):
    """Uploads a single file to GCS."""
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(source_file)
    logger.info(f"✅ Uploaded: {source_file.name} → gs://{BUCKET_NAME}/{destination_blob}")


def upload_debiased_data():
    """Uploads all debiased data files to GCS."""
    if not DATA_DIR.exists():
        logger.error(f"❌ Debiased data folder not found: {DATA_DIR}")
        return

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        logger.warning("⚠ No debiased JSON files found to upload.")
        return

    logger.info(f"🚀 Uploading {len(json_files)} files to GCS bucket: {BUCKET_NAME}")
    for file_path in json_files:
        destination_blob = f"{GCS_FOLDER}/{file_path.name}"
        upload_file_to_gcs(bucket, file_path, destination_blob)

    logger.info("✨ All debiased files uploaded successfully.")


if __name__ == "__main__":
    upload_debiased_data()