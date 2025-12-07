# Model_Pipeline/src/data/load_from_gcs.py
from google.cloud import storage
from google.oauth2 import service_account
import json
import logging
from typing import List, Dict, Optional, Tuple, Union
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve default credentials relative to the repository root (Intelligent-Onboarding-Assistant)
# This goes up 4 levels from: Model_Pipeline/src/data/load_from_gcs.py -> Intelligent-Onboarding-Assistant
DEFAULT_CREDENTIALS_REL_PATH = (
    Path(__file__).resolve().parents[3] / "mlops-476419-2c1937dab204.json"
)


class GCSDataLoader:
    def __init__(
        self,
        bucket_name: str,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        local_cache_dir: Optional[str] = None,
        peek_bytes: int = 8192,
    ):
        """
        peek_bytes: how many bytes to download when peeking to decide if a blob contains JSON.
        """
        self.bucket_name = bucket_name
        self.local_cache_dir = local_cache_dir
        self.peek_bytes = peek_bytes
        if local_cache_dir:
            os.makedirs(local_cache_dir, exist_ok=True)

        # Resolve credentials: explicit argument -> env var -> default file -> ADC
        resolved_creds_path: Optional[Path] = None
        if credentials_path:
            resolved_creds_path = Path(credentials_path).expanduser()
        else:
            env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if env_path:
                resolved_creds_path = Path(env_path).expanduser()
            elif DEFAULT_CREDENTIALS_REL_PATH.exists():
                resolved_creds_path = DEFAULT_CREDENTIALS_REL_PATH

        if resolved_creds_path is not None and resolved_creds_path.exists():
            logger.info(f"Using credentials file: {resolved_creds_path}")
            creds = service_account.Credentials.from_service_account_file(str(resolved_creds_path))
            self.client = storage.Client(project=project_id, credentials=creds)
        else:
            if resolved_creds_path is not None:
                logger.warning(f"Credentials path provided but not found: {resolved_creds_path}. Falling back to ADC.")
            else:
                logger.info("No credentials file provided; using Application Default Credentials (ADC).")
            self.client = storage.Client(project=project_id)

        try:
            self.bucket = self.client.bucket(bucket_name)
            logger.info(f"Initialized GCS client for bucket: {bucket_name}")
        except Exception as e:
            logger.exception("Failed to initialize GCS client or bucket. Check credentials/permissions.")
            raise

    def _download_blob_text_with_retries(self, blob, retries: int = 3, backoff: float = 1.0) -> str:
        for attempt in range(1, retries + 1):
            try:
                # use download_as_text for easier handling of text-based JSON content
                return blob.download_as_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed to download {blob.name}: {e}")
                if attempt == retries:
                    logger.exception(f"Giving up downloading {blob.name}")
                    raise
                time.sleep(backoff * attempt)
        raise RuntimeError("unreachable")

    def _download_blob_prefix(self, blob, nbytes: int) -> bytes:
        """
        Download the first nbytes of the blob. Uses the range header via download_as_bytes with start/ end.
        """
        try:
            # Cloud Storage Python client supports start/end via Blob.download_as_bytes(start=..., end=...)
            # end is inclusive, so request nbytes-1
            return blob.download_as_bytes(start=0, end=max(0, nbytes - 1))
        except Exception:
            # Fallback: full download (will be handled by caller but we try to avoid this)
            logger.debug(f"Prefix download failed for {blob.name}, falling back to full download.")
            return blob.download_as_bytes()

    def _looks_like_json(self, sample_bytes: Union[bytes, str]) -> bool:
        """
        Heuristic to decide if a bytes buffer likely contains JSON or JSONL text.
        We decode to utf-8 ignoring errors, then check first non-whitespace char,
        or check if many lines look like JSON objects (JSONL).
        """
        if isinstance(sample_bytes, bytes):
            try:
                text = sample_bytes.decode("utf-8", errors="ignore")
            except Exception:
                return False
        else:
            text = str(sample_bytes)

        s = text.lstrip()
        if not s:
            return False

        first = s[0]
        if first in ("[", "{"):
            return True

        # JSON Lines heuristic: many lines starting with '{' or '['
        lines = [line.strip() for line in s.splitlines() if line.strip()]
        if not lines:
            return False

        # if first non-empty lines start with { or [, odds are this is JSONL
        score = 0
        check_n = min(5, len(lines))
        for i in range(check_n):
            if lines[i][0] in ("{", "["):
                score += 1
        return score >= 1  # be permissive: even one JSON-like line is enough

    def _parse_json_content(self, text: str, filename: str) -> List[Dict]:
        """
        Correctly parse JSON arrays, even large ones.
        Falls back to JSONL only if array/object parsing fails.
        """
        text = text.strip()
        if not text:
            return []

        # --- first, try strict JSON (full file) ---
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception as e:
            logger.warning(f"Normal JSON parse failed for {filename}: {e}")

        # --- fallback: JSON Lines ---
        records = []
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                logger.warning(f"Failed parsing line {i+1} in {filename} as JSON")
        return records

    def load_chunked_data(self, prefix: str, max_files: Optional[int] = None, extension: Optional[str] = ".json") -> Tuple[List[Dict], Dict[str,int]]:
        """
        Load JSON files from GCS prefix.

        Behavior changes compared to earlier:
        - If 'extension' is provided, blobs whose names end with that extension are preferred.
        - For other blobs (e.g., .pdf but containing JSON text), we peek at the first few KB
          and attempt to detect JSON/JSONL. If the peek indicates JSON, we download & parse it.
        - This allows loading JSON stored inside blobs with non-.json filenames (e.g. PDFs containing JSON).
        """
        chunks: List[Dict] = []
        per_file_counts: Dict[str, int] = {}

        blobs = self.bucket.list_blobs(prefix=prefix)
        file_count = 0

        for blob in blobs:
            if max_files is not None and file_count >= max_files:
                break

            # skip "directory" markers
            if blob.name.endswith("/"):
                continue

            logger.info(f"Considering GCS object: {blob.name}")

            # Quick skip by extension if provided: prefer matching names (but still may peek into others)
            name_matches_ext = (extension is None) or blob.name.endswith(extension)

            # If we have a local cache file, prefer it (but only if it exists)
            content = None
            local_path = None
            if self.local_cache_dir:
                rel_path = blob.name.replace("/", "_")
                local_path = Path(self.local_cache_dir) / rel_path
                if local_path.exists():
                    logger.debug(f"Using cached file {local_path}")
                    content = local_path.read_text(encoding="utf-8")

            # If we already got content from cache, parse it
            if content is not None:
                records = self._parse_json_content(content, blob.name)
                per_file_counts[blob.name] = len(records)
                if records:
                    chunks.extend(records)
                logger.info(f"  -> Parsed {len(records)} records from cached {blob.name}")
                file_count += 1
                continue

            # If filename matches desired extension, download whole object and parse
            try_full_download = False
            if name_matches_ext:
                try_full_download = True
            else:
                # Peek at first few KB to see if it "looks like" JSON/JSONL
                try:
                    prefix_bytes = self._download_blob_prefix(blob, self.peek_bytes)
                    if self._looks_like_json(prefix_bytes):
                        logger.info(f"  -> Peek detected JSON-like content in {blob.name}; will download full text")
                        try_full_download = True
                    else:
                        logger.debug(f"  -> Peek did not detect JSON-like content for {blob.name}; skipping")
                        try_full_download = False
                except Exception as e:
                    logger.warning(f"Failed to peek into {blob.name}: {e}. Will skip this file.")
                    try_full_download = False

            if not try_full_download:
                # skip blob
                continue

            # Download full content (with retries)
            try:
                content = self._download_blob_text_with_retries(blob)
            except Exception:
                logger.exception(f"Failed to download {blob.name}; skipping.")
                continue

            # Save to local cache if requested
            if self.local_cache_dir and local_path is not None:
                try:
                    local_path.write_text(content, encoding="utf-8")
                except Exception:
                    logger.debug(f"Failed to write cache for {local_path}")

            # parse into records
            records = self._parse_json_content(content, blob.name)
            per_file_counts[blob.name] = len(records)
            if records:
                chunks.extend(records)

            logger.info(f"  -> Parsed {len(records)} records from {blob.name}")

            file_count += 1

        logger.info(f"Loaded {len(chunks)} total records from GCS prefix={prefix}")
        return chunks, per_file_counts

    # --- Important: use the exact filenames that you uploaded to GCS ---
    def load_handbook_chunks(self, max_files: Optional[int] = None) -> Tuple[List[Dict], Dict[str,int]]:
        """
        Load the handbook JSON file from GCS (explicit filename).
        """
        return self.load_chunked_data(prefix="onboarding_ai/debiased_data/handbook_paragraphs_debiased.json", max_files=max_files)

    def load_transcript_chunks(self, max_files: Optional[int] = None) -> Tuple[List[Dict], Dict[str,int]]:
        """
        Load the transcripts JSON file from GCS (explicit filename).
        """
        return self.load_chunked_data(prefix="onboarding_ai/debiased_data/all_transcripts_debiased.json", max_files=max_files)

    def load_all_chunks(self, max_files_per_prefix: Optional[int] = None) -> Tuple[List[Dict], Dict[str,int]]:
        """
        Load both handbook and transcript chunks, dedupe, and return combined list.
        Returns (all_chunks, aggregated_per_file_counts)
        """
        logger.info("Loading all chunks from GCS...")

        handbook_list, handbook_counts = self.load_handbook_chunks(max_files=max_files_per_prefix)
        transcripts_list, transcript_counts = self.load_transcript_chunks(max_files=max_files_per_prefix)

        # aggregate per-file counts
        aggregated_counts = {}
        aggregated_counts.update(handbook_counts)
        aggregated_counts.update(transcript_counts)

        # combine and dedupe by id if present, otherwise by title+paragraph hash
        all_combined = []
        seen = set()
        for chunk in handbook_list + transcripts_list:
            # create stable key
            key = None
            if isinstance(chunk, dict):
                for id_key in ("id", "uid", "doc_id"):
                    if id_key in chunk:
                        key = f"{id_key}:{chunk[id_key]}"
                        break
                if key is None:
                    title = chunk.get("title", "")
                    para = chunk.get("paragraph", "")
                    key = f"HASH:{hash((title.strip(), para.strip()))}"
            else:
                key = f"HASH:{hash(str(chunk))}"

            if key in seen:
                continue
            seen.add(key)
            all_combined.append(chunk)

        logger.info(f"Total chunks loaded (deduped): {len(all_combined)}")
        return all_combined, aggregated_counts


if __name__ == "__main__":
    # Quick test / CLI: uses the default credential path if present (resolved relative to this file)
    BUCKET_NAME = "mlops-data-oa"
    PROJECT_ID = os.environ.get("GCP_PROJECT")
    # Force using default credentials candidate if present
    creds_env = os.environ.get("MY_GCS_CREDS")  # optional custom env var
    creds_path = creds_env if creds_env else (str(DEFAULT_CREDENTIALS_REL_PATH) if DEFAULT_CREDENTIALS_REL_PATH.exists() else None)

    logger.info(f"Resolved default credentials candidate: {DEFAULT_CREDENTIALS_REL_PATH}")
    if creds_path:
        logger.info(f"Credentials path used for this run: {creds_path}")
    else:
        logger.info("No credentials path detected; will try ADC if available.")

    loader = GCSDataLoader(bucket_name=BUCKET_NAME, project_id=PROJECT_ID, credentials_path=creds_path, local_cache_dir="./.gcs_cache")

    all_chunks, counts = loader.load_all_chunks(max_files_per_prefix=20)
    # Print per-file counts
    logger.info("Per-file counts from GCS:")
    for fname, c in counts.items():
        logger.info(f"  - {fname}: {c} records")
    print(f"\nTotal unique chunks (deduped): {len(all_chunks)}")
    if all_chunks:
        print("\nSample record:")
        print(json.dumps(all_chunks[0], indent=2))
