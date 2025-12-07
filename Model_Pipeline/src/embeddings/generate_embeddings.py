from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
from google.cloud import storage
from google.oauth2 import service_account
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, 
                 model_name: str = "all-mpnet-base-v2", 
                 use_gcs: bool = True,
                 bucket_name: str = "mlops-data-oa",
                 project_id: str = "mlops-476419",
                 credentials_path: str = None):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.use_gcs = use_gcs
        self.bucket_name = bucket_name
        
        # Initialize GCS client if needed
        if self.use_gcs:
            # Resolve credentials path
            if credentials_path is None:
                # Try environment variable first
                credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                
                # If not in env, use default relative path
                if credentials_path is None:
                    default_creds_path = Path(__file__).resolve().parents[3] / "mlops-476419-2c1937dab204.json"
                    if default_creds_path.exists():
                        credentials_path = str(default_creds_path)
            
            # Initialize GCS client with credentials
            if credentials_path and Path(credentials_path).exists():
                logger.info(f"Using credentials file: {credentials_path}")
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.gcs_client = storage.Client(project=project_id, credentials=credentials)
            else:
                logger.warning("No credentials found, trying Application Default Credentials")
                self.gcs_client = storage.Client(project=project_id)
            
            self.bucket = self.gcs_client.bucket(bucket_name)
            logger.info(f"GCS client initialized for bucket: {bucket_name}")
        
        logger.info(f"Model loaded! Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  
        )
        
        logger.info(f"✓ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def process_chunks(self, chunks: List[Dict], text_field: str = "paragraph") -> Tuple[List[str], np.ndarray, List[Dict]]:
        logger.info(f"Processing {len(chunks)} chunks...")
        
        texts = []
        metadata = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.get(text_field, chunk.get('text', chunk.get('content', '')))
            
            if not text or not text.strip():
                logger.warning(f"Empty text in chunk {i}, skipping...")
                continue
            
            texts.append(text)
            
            meta = {
                'chunk_id': chunk.get('chunk_id', f'chunk_{i}'),
                'title': chunk.get('title', ''),
                'source': chunk.get('source', 'unknown'),
                'source_type': chunk.get('source_type', 'unknown'),
                'url': chunk.get('url', ''),
                'original_index': i
            }
            metadata.append(meta)
        
        logger.info(f"Extracted {len(texts)} valid texts")
        
        embeddings = self.generate_embeddings(texts)
        
        return texts, embeddings, metadata
    
    def save_embeddings(self, 
                       texts: List[str],
                       embeddings: np.ndarray, 
                       metadata: List[Dict], 
                       output_dir: str = "models/embeddings",
                       gcs_prefix: str = "onboarding_ai/embeddings/"):
        """
        Save embeddings to local directory and optionally to GCS
        
        Args:
            texts: List of text strings
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
            output_dir: Local directory path (for local storage or temp files)
            gcs_prefix: GCS prefix path (used if use_gcs=True)
        """
        # Always create local temp directory for intermediate storage
        os.makedirs(output_dir, exist_ok=True)
        
        # Save embeddings locally first
        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_path, embeddings)
        logger.info(f"✓ Saved embeddings locally to: {embeddings_path}")
        
        texts_path = os.path.join(output_dir, "texts.json")
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved texts locally to: {texts_path}")
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved metadata locally to: {metadata_path}")
        
        model_info = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'num_embeddings': len(embeddings),
            'num_texts': len(texts)
        }
        info_path = os.path.join(output_dir, "model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"✓ Saved model info locally to: {info_path}")
        
        # Upload to GCS if enabled
        if self.use_gcs:
            logger.info(f"\nUploading embeddings to GCS bucket: {self.bucket_name}")
            self._upload_to_gcs(embeddings_path, f"{gcs_prefix}embeddings.npy")
            self._upload_to_gcs(texts_path, f"{gcs_prefix}texts.json")
            self._upload_to_gcs(metadata_path, f"{gcs_prefix}metadata.json")
            self._upload_to_gcs(info_path, f"{gcs_prefix}model_info.json")
            logger.info(f"✓ All files uploaded to GCS: gs://{self.bucket_name}/{gcs_prefix}")
        
        logger.info(f"\n✓ All files saved successfully!")
    
    def _upload_to_gcs(self, local_path: str, gcs_path: str):
        """Upload a file to GCS"""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            logger.info(f"  ✓ Uploaded: {gcs_path}")
        except Exception as e:
            logger.error(f"  ✗ Failed to upload {gcs_path}: {e}")
            raise
    
    def _download_from_gcs(self, gcs_path: str, local_path: str):
        """Download a file from GCS"""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.download_to_filename(local_path)
            logger.info(f"  ✓ Downloaded: {gcs_path}")
        except Exception as e:
            logger.error(f"  ✗ Failed to download {gcs_path}: {e}")
            raise
    
    def load_embeddings(self, 
                       input_dir: str = "models/embeddings",
                       gcs_prefix: str = "onboarding_ai/embeddings/") -> Tuple[List[str], np.ndarray, List[Dict]]:
        """
        Load embeddings from GCS (if enabled) or local directory
        
        Args:
            input_dir: Local directory path
            gcs_prefix: GCS prefix path (used if use_gcs=True)
        
        Returns:
            Tuple of (texts, embeddings, metadata)
        """
        # If using GCS, download files first
        if self.use_gcs:
            logger.info(f"Downloading embeddings from GCS: gs://{self.bucket_name}/{gcs_prefix}")
            os.makedirs(input_dir, exist_ok=True)
            
            self._download_from_gcs(
                f"{gcs_prefix}embeddings.npy",
                os.path.join(input_dir, "embeddings.npy")
            )
            self._download_from_gcs(
                f"{gcs_prefix}texts.json",
                os.path.join(input_dir, "texts.json")
            )
            self._download_from_gcs(
                f"{gcs_prefix}metadata.json",
                os.path.join(input_dir, "metadata.json")
            )
        
        # Load from local directory
        embeddings_path = os.path.join(input_dir, "embeddings.npy")
        texts_path = os.path.join(input_dir, "texts.json")
        metadata_path = os.path.join(input_dir, "metadata.json")
        
        embeddings = np.load(embeddings_path)
        
        with open(texts_path, 'r', encoding='utf-8') as f:
            texts = json.load(f)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"✓ Loaded {len(embeddings)} embeddings")
        return texts, embeddings, metadata


if __name__ == "__main__":
    # Import GCS data loader
    from data.load_from_gcs import GCSDataLoader
    
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data from GCS...")
    logger.info("=" * 60)
    
    # Get credentials path
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path is None:
        default_creds = Path(__file__).resolve().parents[3] / "mlops-476419-2c1937dab204.json"
        if default_creds.exists():
            creds_path = str(default_creds)
    
    # Initialize GCS data loader
    gcs_loader = GCSDataLoader(
        bucket_name="mlops-data-oa",
        project_id="mlops-476419",
        credentials_path=creds_path,
        local_cache_dir="./.gcs_cache"
    )
    
    # Load all chunks from GCS
    chunks, file_counts = gcs_loader.load_all_chunks(max_files_per_prefix=None)
    
    logger.info(f"\nLoaded {len(chunks)} total chunks from GCS")
    logger.info("Per-file counts:")
    for fname, count in file_counts.items():
        logger.info(f"  - {fname}: {count} chunks")
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Generating embeddings...")
    logger.info("=" * 60)
    
    # Initialize embedding generator with GCS support
    embedding_gen = EmbeddingGenerator(
        model_name="all-mpnet-base-v2",
        use_gcs=True,  # Enable GCS upload
        bucket_name="mlops-data-oa",
        project_id="mlops-476419",
        credentials_path=creds_path  # Pass credentials path
    )
    
    # Process chunks and generate embeddings
    texts, embeddings, metadata = embedding_gen.process_chunks(chunks)
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Saving embeddings...")
    logger.info("=" * 60)
    
    # Save embeddings locally and to GCS
    embedding_gen.save_embeddings(
        texts, 
        embeddings, 
        metadata,
        output_dir="models/embeddings",  # Local temp directory
        gcs_prefix="onboarding_ai/embeddings/"  # GCS path
    )
    
    print("\n" + "=" * 60)
    print("✓ EMBEDDING GENERATION COMPLETE!")
    print("=" * 60)
    print(f"✓ Total chunks processed: {len(texts)}")
    print(f"✓ Embedding dimensions: {embeddings.shape}")
    print(f"✓ Local files: models/embeddings/")
    print(f"✓ GCS location: gs://mlops-data-oa/onboarding_ai/embeddings/")
    print("=" * 60)
    
    # Optional: Test loading embeddings back
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Testing embedding load from GCS...")
    logger.info("=" * 60)
    
    loaded_texts, loaded_embeddings, loaded_metadata = embedding_gen.load_embeddings(
        input_dir="models/embeddings_test",  # Different directory to test GCS download
        gcs_prefix="onboarding_ai/embeddings/"
    )
    
    print(f"\n✓ Successfully loaded {len(loaded_embeddings)} embeddings from GCS")
    print(f"✓ Shapes match: {embeddings.shape == loaded_embeddings.shape}")