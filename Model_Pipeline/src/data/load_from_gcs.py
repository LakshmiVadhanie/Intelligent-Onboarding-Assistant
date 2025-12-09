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


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Chunk text into smaller pieces for better RAG performance.
    Uses character-based splitting with overlap to maintain context.
    
    Args:
        text: Text to chunk
        chunk_size: Target size in characters (~200 tokens = ~800 chars)
        overlap: Overlap between chunks in characters
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    # Split on paragraph boundaries first, then sentences
    separators = ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
    
    while start < len(text):
        end = start + chunk_size
        
        # If this is the last chunk, take everything
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        
        # Try to break at a natural boundary
        chunk_end = end
        for sep in separators:
            # Look for separator near the end
            sep_pos = text.rfind(sep, start, end)
            if sep_pos > start + chunk_size // 2:  # Found good break point
                chunk_end = sep_pos + len(sep)
                break
        
        chunks.append(text[start:chunk_end].strip())
        
        # Move start position with overlap
        start = chunk_end - overlap
        if start <= 0 or start >= len(text):
            break
    
    return [c for c in chunks if c]  # Remove empty chunks


class GCSDataLoader:
    def __init__(
        self,
        bucket_name: str,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        local_cache_dir: Optional[str] = None,
        peek_bytes: int = 8192,
        enable_chunking: bool = True,
        handbook_chunk_size: int = 800,
        transcript_chunk_size: int = 800,
        handbook_overlap: int = 150,
        transcript_overlap: int = 100,
    ):
        """
        peek_bytes: how many bytes to download when peeking to decide if a blob contains JSON.
        enable_chunking: Whether to chunk large documents into smaller pieces
        """
        self.bucket_name = bucket_name
        self.local_cache_dir = local_cache_dir
        self.peek_bytes = peek_bytes
        self.enable_chunking = enable_chunking
        self.handbook_chunk_size = handbook_chunk_size
        self.transcript_chunk_size = transcript_chunk_size
        self.handbook_overlap = handbook_overlap
        self.transcript_overlap = transcript_overlap
        
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
            if self.enable_chunking:
                logger.info(f"✓ Chunking enabled: Handbook={handbook_chunk_size}±{handbook_overlap}, Transcripts={transcript_chunk_size}±{transcript_overlap}")
        except Exception as e:
            logger.exception("Failed to initialize GCS client or bucket. Check credentials/permissions.")
            raise

    def _download_blob_text_with_retries(self, blob, retries: int = 3, backoff: float = 1.0) -> str:
        for attempt in range(1, retries + 1):
            try:
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
            return blob.download_as_bytes(start=0, end=max(0, nbytes - 1))
        except Exception:
            logger.debug(f"Prefix download failed for {blob.name}, falling back to full download.")
            return blob.download_as_bytes()

    def _looks_like_json(self, sample_bytes: Union[bytes, str]) -> bool:
        """
        Heuristic to decide if a bytes buffer likely contains JSON or JSONL text.
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

        lines = [line.strip() for line in s.splitlines() if line.strip()]
        if not lines:
            return False

        score = 0
        check_n = min(5, len(lines))
        for i in range(check_n):
            if lines[i][0] in ("{", "["):
                score += 1
        return score >= 1

    def _parse_json_content(self, text: str, filename: str) -> List[Dict]:
        """
        Correctly parse JSON arrays, even large ones.
        Falls back to JSONL only if array/object parsing fails.
        """
        text = text.strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception as e:
            logger.warning(f"Normal JSON parse failed for {filename}: {e}")

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

    def _chunk_records(self, records: List[Dict], source_type: str) -> List[Dict]:
        """
        Chunk records based on their text content.
        
        Args:
            records: List of document records
            source_type: 'handbook' or 'transcript'
        
        Returns:
            List of chunked records with updated metadata
        """
        if not self.enable_chunking:
            return records
        
        # Determine chunk parameters based on source type
        if source_type == 'handbook':
            chunk_size = self.handbook_chunk_size
            overlap = self.handbook_overlap
        else:  # transcript
            chunk_size = self.transcript_chunk_size
            overlap = self.transcript_overlap
        
        chunked_records = []
        record_counter = 0  # Global counter for unique IDs
        
        for record in records:
            # Extract text - try common field names
            text = record.get('text') or record.get('paragraph') or record.get('content') or ''
            
            if not text or len(text) <= chunk_size:
                # Small enough, keep as-is but ensure it has an ID
                if not record.get('id'):
                    record['id'] = f"{source_type}_{record_counter}"
                    record_counter += 1
                # Add source_type field
                record['source_type'] = source_type
                chunked_records.append(record)
                continue
            
            # Chunk the text
            text_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            
            # Create new records for each chunk
            for i, chunk in enumerate(text_chunks):
                chunked_record = record.copy()
                chunked_record['text'] = chunk
                chunked_record['paragraph'] = chunk
                
                # Create UNIQUE ID using source type, counter, and chunk index
                chunked_record['id'] = f"{source_type}_{record_counter}_{i}_{hash(chunk[:50]) % 10000}"
                chunked_record['chunk_index'] = i
                chunked_record['total_chunks'] = len(text_chunks)
                chunked_record['is_chunked'] = True
                chunked_record['source_type'] = source_type
                
                chunked_records.append(chunked_record)
            
            record_counter += 1
            logger.debug(f"Chunked '{record.get('title', 'unknown')}' into {len(text_chunks)} pieces")
        
        return chunked_records

    def load_chunked_data(self, prefix: str, source_type: str = 'unknown', max_files: Optional[int] = None, extension: Optional[str] = ".json") -> Tuple[List[Dict], Dict[str,int]]:
        """
        Load JSON files from GCS prefix and optionally chunk them.
        """
        chunks: List[Dict] = []
        per_file_counts: Dict[str, int] = {}

        blobs = self.bucket.list_blobs(prefix=prefix)
        file_count = 0

        for blob in blobs:
            if max_files is not None and file_count >= max_files:
                break

            if blob.name.endswith("/"):
                continue

            logger.info(f"Considering GCS object: {blob.name}")

            name_matches_ext = (extension is None) or blob.name.endswith(extension)

            content = None
            local_path = None
            if self.local_cache_dir:
                rel_path = blob.name.replace("/", "_")
                local_path = Path(self.local_cache_dir) / rel_path
                if local_path.exists():
                    logger.debug(f"Using cached file {local_path}")
                    content = local_path.read_text(encoding="utf-8")

            if content is not None:
                records = self._parse_json_content(content, blob.name)
                records = self._chunk_records(records, source_type)
                per_file_counts[blob.name] = len(records)
                if records:
                    chunks.extend(records)
                logger.info(f"  -> Parsed and chunked into {len(records)} pieces from cached {blob.name}")
                file_count += 1
                continue

            try_full_download = False
            if name_matches_ext:
                try_full_download = True
            else:
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
                continue

            try:
                content = self._download_blob_text_with_retries(blob)
            except Exception:
                logger.exception(f"Failed to download {blob.name}; skipping.")
                continue

            if self.local_cache_dir and local_path is not None:
                try:
                    local_path.write_text(content, encoding="utf-8")
                except Exception:
                    logger.debug(f"Failed to write cache for {local_path}")

            records = self._parse_json_content(content, blob.name)
            records = self._chunk_records(records, source_type)
            per_file_counts[blob.name] = len(records)
            if records:
                chunks.extend(records)

            logger.info(f"  -> Parsed and chunked into {len(records)} pieces from {blob.name}")

            file_count += 1

        logger.info(f"Loaded {len(chunks)} total chunked records from GCS prefix={prefix}")
        return chunks, per_file_counts

    def load_handbook_chunks(self, max_files: Optional[int] = None) -> Tuple[List[Dict], Dict[str,int]]:
        """
        Load the handbook JSON file from GCS (explicit filename) and chunk it.
        """
        return self.load_chunked_data(
            prefix="onboarding_ai/debiased_data/handbook_paragraphs_debiased.json", 
            source_type='handbook',
            max_files=max_files
        )

    def load_transcript_chunks(self, max_files: Optional[int] = None) -> Tuple[List[Dict], Dict[str,int]]:
        """
        Load the transcripts JSON file from GCS (explicit filename) and chunk it.
        """
        return self.load_chunked_data(
            prefix="onboarding_ai/debiased_data/all_transcripts_debiased.json", 
            source_type='transcript',
            max_files=max_files
        )

    def load_all_chunks(self, max_files_per_prefix: Optional[int] = None) -> Tuple[List[Dict], Dict[str,int]]:
        """
        Load both handbook and transcript chunks, chunk them, dedupe, and return combined list.
        Returns (all_chunks, aggregated_per_file_counts)
        """
        logger.info("Loading all chunks from GCS...")

        handbook_list, handbook_counts = self.load_handbook_chunks(max_files=max_files_per_prefix)
        transcripts_list, transcript_counts = self.load_transcript_chunks(max_files=max_files_per_prefix)

        aggregated_counts = {}
        aggregated_counts.update(handbook_counts)
        aggregated_counts.update(transcript_counts)

        all_combined = []
        seen = set()
        
        for chunk in handbook_list + transcripts_list:
            if not isinstance(chunk, dict):
                continue
            
            chunk_id = chunk.get('id', '')
            
            if not chunk_id:
                title = chunk.get("title", "")
                text = chunk.get("paragraph") or chunk.get("text", "")
                text_sample = text[:200].strip() if text else ""
                chunk_id = f"HASH_{hash((title.strip(), text_sample))}"
            
            if chunk_id in seen:
                logger.debug(f"Skipping duplicate chunk: {chunk_id}")
                continue
            
            seen.add(chunk_id)
            all_combined.append(chunk)

        logger.info(f"Total chunks loaded: {len(all_combined)} (after deduplication)")
        
        if self.enable_chunking:
            chunked_count = sum(1 for c in all_combined if c.get('is_chunked', False))
            logger.info(f"  ✓ {chunked_count} chunks were created from larger documents")
        
        handbook_final = sum(1 for c in all_combined if c.get('source_type') == 'handbook')
        transcript_final = sum(1 for c in all_combined if c.get('source_type') == 'transcript')
        logger.info(f"  ✓ Breakdown: {handbook_final} handbook + {transcript_final} transcripts = {len(all_combined)} total")
        
        return all_combined, aggregated_counts


if __name__ == "__main__":
    BUCKET_NAME = "mlops-data-oa"
    PROJECT_ID = os.environ.get("GCP_PROJECT")
    creds_env = os.environ.get("MY_GCS_CREDS")
    creds_path = creds_env if creds_env else (str(DEFAULT_CREDENTIALS_REL_PATH) if DEFAULT_CREDENTIALS_REL_PATH.exists() else None)

    logger.info(f"Resolved default credentials candidate: {DEFAULT_CREDENTIALS_REL_PATH}")
    if creds_path:
        logger.info(f"Credentials path used for this run: {creds_path}")
    else:
        logger.info("No credentials path detected; will try ADC if available.")

    loader = GCSDataLoader(
        bucket_name=BUCKET_NAME, 
        project_id=PROJECT_ID, 
        credentials_path=creds_path, 
        local_cache_dir="./.gcs_cache",
        enable_chunking=True,
        handbook_chunk_size=800,
        transcript_chunk_size=800,
        handbook_overlap=150,
        transcript_overlap=100
    )

    all_chunks, counts = loader.load_all_chunks(max_files_per_prefix=20)
    logger.info("Per-file counts from GCS:")
    for fname, c in counts.items():
        logger.info(f"  - {fname}: {c} records (after chunking)")
    print(f"\nTotal unique chunks (deduped, after chunking): {len(all_chunks)}")
    
    chunked = [c for c in all_chunks if c.get('is_chunked', False)]
    print(f"Documents that were chunked: {len(chunked)}")
    
    if all_chunks:
        print("\nSample record:")
        print(json.dumps(all_chunks[0], indent=2, ensure_ascii=False)[:500] + "...")
        
        if chunked:
            print("\nSample chunked record:")
            print(json.dumps(chunked[0], indent=2, ensure_ascii=False)[:500] + "...")