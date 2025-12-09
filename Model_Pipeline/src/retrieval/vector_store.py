import chromadb
from chromadb.config import Settings
import numpy as np
import json
from typing import List, Dict, Optional
import logging
from pathlib import Path
from google.cloud import storage
from google.oauth2 import service_account
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, 
                 collection_name: str = "gitlab_onboarding",
                 persist_directory: str = "models/vector_store",
                 use_gcs: bool = False,
                 bucket_name: str = "mlops-data-oa",
                 project_id: str = "mlops-476419",
                 credentials_path: str = None):
        """
        Initialize VectorStore with optional GCS support
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Local directory for ChromaDB persistence
            use_gcs: Whether to sync with GCS
            bucket_name: GCS bucket name
            project_id: GCP project ID
            credentials_path: Path to service account credentials
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_gcs = use_gcs
        self.bucket_name = bucket_name
        
        # Create local directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize GCS client if needed
        if self.use_gcs:
            if credentials_path is None:
                credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if credentials_path is None:
                    default_creds = Path(__file__).resolve().parents[3] / "mlops-476419-2c1937dab204.json"
                    if default_creds.exists():
                        credentials_path = str(default_creds)
            
            if credentials_path and Path(credentials_path).exists():
                logger.info(f"Using GCS credentials: {credentials_path}")
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.gcs_client = storage.Client(project=project_id, credentials=credentials)
            else:
                self.gcs_client = storage.Client(project=project_id)
            
            self.bucket = self.gcs_client.bucket(bucket_name)
            logger.info(f"✓ GCS client initialized for bucket: {bucket_name}")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  
        )
        
        logger.info(f"✓ Initialized VectorStore: {collection_name}")
        logger.info(f"✓ Persist directory: {persist_directory}")
        logger.info(f"✓ Current documents in collection: {self.collection.count()}")
    
    def add_documents(self, 
                     texts: List[str],
                     embeddings: np.ndarray,
                     metadatas: List[Dict],
                     ids: Optional[List[str]] = None):
        """Add documents to the vector store"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        embeddings_list = embeddings.tolist()
        
        logger.info(f"Adding {len(texts)} documents to collection...")
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"✓ Successfully added {len(texts)} documents")
        logger.info(f"✓ Total documents in collection: {self.collection.count()}")
    
    def query(self, 
             query_embedding: np.ndarray,
             n_results: int = 5,
             where: Optional[Dict] = None) -> Dict:
        """Query the vector store"""
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def get_all_documents(self) -> Dict:
        """Get all documents from collection"""
        return self.collection.get()
    
    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"✓ Deleted collection: {self.collection_name}")
    
    def persist(self):
        """Persist the vector store"""
        logger.info(f"✓ Vector store persisted to: {self.persist_directory}")
    
    def _download_from_gcs(self, gcs_path: str, local_path: str):
        """Download file from GCS"""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.download_to_filename(local_path)
            logger.info(f"  ✓ Downloaded from GCS: {gcs_path}")
        except Exception as e:
            logger.error(f"  ✗ Failed to download {gcs_path}: {e}")
            raise


def load_and_index_embeddings(embeddings_dir: str = "models/embeddings",
                              vector_store_dir: str = "models/vector_store",
                              use_gcs: bool = True,
                              gcs_embeddings_prefix: str = "onboarding_ai/embeddings/",
                              bucket_name: str = "mlops-data-oa",
                              project_id: str = "mlops-476419",
                              credentials_path: str = None):
    """
    Load embeddings from GCS (if enabled) and index into ChromaDB
    
    Args:
        embeddings_dir: Local directory for embeddings
        vector_store_dir: Directory for ChromaDB persistence
        use_gcs: Whether to download from GCS first
        gcs_embeddings_prefix: GCS prefix for embeddings
        bucket_name: GCS bucket name
        project_id: GCP project ID
        credentials_path: Path to service account credentials
    """
    logger.info("=" * 60)
    logger.info("LOADING AND INDEXING EMBEDDINGS")
    logger.info("=" * 60)
    
    # Download from GCS if enabled
    if use_gcs:
        logger.info(f"Downloading embeddings from GCS: gs://{bucket_name}/{gcs_embeddings_prefix}")
        
        # Initialize GCS client
        if credentials_path is None:
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_path is None:
                default_creds = Path(__file__).resolve().parents[3] / "mlops-476419-2c1937dab204.json"
                if default_creds.exists():
                    credentials_path = str(default_creds)
        
        if credentials_path and Path(credentials_path).exists():
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            gcs_client = storage.Client(project=project_id, credentials=credentials)
        else:
            gcs_client = storage.Client(project=project_id)
        
        bucket = gcs_client.bucket(bucket_name)
        
        # Create local directory
        Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
        
        # Download files
        files_to_download = [
            ("embeddings.npy", "embeddings.npy"),
            ("texts.json", "texts.json"),
            ("metadata.json", "metadata.json"),
            ("model_info.json", "model_info.json")
        ]
        
        for gcs_filename, local_filename in files_to_download:
            gcs_path = f"{gcs_embeddings_prefix}{gcs_filename}"
            local_path = Path(embeddings_dir) / local_filename
            
            try:
                blob = bucket.blob(gcs_path)
                blob.download_to_filename(str(local_path))
                logger.info(f"  ✓ Downloaded: {gcs_filename}")
            except Exception as e:
                logger.warning(f"  ✗ Failed to download {gcs_filename}: {e}")
    
    # Load embeddings from local directory
    logger.info(f"\n📂 Loading from: {embeddings_dir}")
    
    embeddings_path = Path(embeddings_dir) / "embeddings.npy"
    texts_path = Path(embeddings_dir) / "texts.json"
    metadata_path = Path(embeddings_dir) / "metadata.json"
    
    embeddings = np.load(embeddings_path)
    logger.info(f"✓ Loaded embeddings: {embeddings.shape}")
    
    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    logger.info(f"✓ Loaded {len(texts)} texts")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadatas = json.load(f)
    logger.info(f"✓ Loaded {len(metadatas)} metadata entries")
    
    # Create vector store
    logger.info("\n📦 Creating vector store...")
    vector_store = VectorStore(
        collection_name="gitlab_onboarding",
        persist_directory=vector_store_dir,
        use_gcs=use_gcs,
        bucket_name=bucket_name,
        project_id=project_id,
        credentials_path=credentials_path
    )
    
    # Check if collection already has documents
    if vector_store.collection.count() > 0:
        logger.warning(f"⚠ Collection already has {vector_store.collection.count()} documents")
        response = input("Do you want to delete and re-index? (yes/no): ")
        if response.lower() == 'yes':
            vector_store.delete_collection()
            vector_store = VectorStore(
                collection_name="gitlab_onboarding",
                persist_directory=vector_store_dir,
                use_gcs=use_gcs,
                bucket_name=bucket_name,
                project_id=project_id,
                credentials_path=credentials_path
            )
    
    # Index documents
    logger.info("\n📊 Indexing documents...")
    vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    vector_store.persist()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ INDEXING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"✓ Total indexed documents: {vector_store.collection.count()}")
    logger.info(f"✓ Vector store location: {vector_store_dir}")
    if use_gcs:
        logger.info(f"✓ Embeddings source: gs://{bucket_name}/{gcs_embeddings_prefix}")
    logger.info("=" * 60)
    
    return vector_store


if __name__ == "__main__":
    # Get credentials path
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path is None:
        default_creds = Path(__file__).resolve().parents[3] / "mlops-476419-2c1937dab204.json"
        if default_creds.exists():
            creds_path = str(default_creds)
    
    # Load and index with GCS support
    vector_store = load_and_index_embeddings(
        embeddings_dir="models/embeddings",
        vector_store_dir="models/vector_store",
        use_gcs=True,  # Enable GCS download
        gcs_embeddings_prefix="onboarding_ai/embeddings/",
        bucket_name="mlops-data-oa",
        project_id="mlops-476419",
        credentials_path=creds_path
    )
    
    print("\n" + "=" * 60)
    print("🧪 TESTING VECTOR STORE")
    print("=" * 60)
    
    # Load embeddings for testing
    embeddings = np.load("models/embeddings/embeddings.npy")
    test_embedding = embeddings[0]  
    
    print(f"\n🔍 Running test query...")
    results = vector_store.query(test_embedding, n_results=3)
    
    print(f"\n✓ Found {len(results['ids'][0])} results:")
    for i, (doc_id, document, distance) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['distances'][0]
    )):
        print(f"\n--- Result {i+1} ---")
        print(f"ID: {doc_id}")
        print(f"Distance: {distance:.4f}")
        print(f"Text preview: {document[:150]}...")
    
    print("\n" + "=" * 60)
    print("✅ VECTOR STORE TEST COMPLETE!")
    print("=" * 60)