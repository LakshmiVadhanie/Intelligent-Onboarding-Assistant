import chromadb
from chromadb.config import Settings
import numpy as np
import json
from typing import List, Dict, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, 
                 collection_name: str = "gitlab_onboarding",
                 persist_directory: str = "models/vector_store"):

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  
        )
        
        logger.info(f" Initialized VectorStore: {collection_name}")
        logger.info(f" Persist directory: {persist_directory}")
        logger.info(f" Current documents in collection: {self.collection.count()}")
    
    def add_documents(self, 
                     texts: List[str],
                     embeddings: np.ndarray,
                     metadatas: List[Dict],
                     ids: Optional[List[str]] = None):

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
        
        logger.info(f" Successfully added {len(texts)} documents")
        logger.info(f" Total documents in collection: {self.collection.count()}")
    
    def query(self, 
             query_embedding: np.ndarray,
             n_results: int = 5,
             where: Optional[Dict] = None) -> Dict:
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def get_all_documents(self) -> Dict:
        return self.collection.get()
    
    def delete_collection(self):
        self.client.delete_collection(name=self.collection_name)
        logger.info(f" Deleted collection: {self.collection_name}")
    
    def persist(self):
        logger.info(f" Vector store persisted to: {self.persist_directory}")


def load_and_index_embeddings(embeddings_dir: str = "models/embeddings",
                              vector_store_dir: str = "models/vector_store"):
    logger.info("=" * 60)
    logger.info("LOADING AND INDEXING EMBEDDINGS")
    logger.info("=" * 60)
    
    logger.info(f" Loading from: {embeddings_dir}")
    
    embeddings_path = Path(embeddings_dir) / "embeddings.npy"
    texts_path = Path(embeddings_dir) / "texts.json"
    metadata_path = Path(embeddings_dir) / "metadata.json"
    
    embeddings = np.load(embeddings_path)
    logger.info(f" Loaded embeddings: {embeddings.shape}")
    
    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    logger.info(f" Loaded {len(texts)} texts")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadatas = json.load(f)
    logger.info(f" Loaded {len(metadatas)} metadata entries")
    
    logger.info("\n Creating vector store...")
    vector_store = VectorStore(
        collection_name="gitlab_onboarding",
        persist_directory=vector_store_dir
    )
    
    if vector_store.collection.count() > 0:
        logger.warning(f" Collection already has {vector_store.collection.count()} documents")
        response = input("Do you want to delete and re-index? (yes/no): ")
        if response.lower() == 'yes':
            vector_store.delete_collection()
            vector_store = VectorStore(
                collection_name="gitlab_onboarding",
                persist_directory=vector_store_dir
            )
    
    logger.info("\n Indexing documents...")
    vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    vector_store.persist()
    
    logger.info("\n" + "=" * 60)
    logger.info(" INDEXING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f" Total indexed documents: {vector_store.collection.count()}")
    logger.info(f" Vector store location: {vector_store_dir}")
    logger.info("=" * 60)
    
    return vector_store


if __name__ == "__main__":
    vector_store = load_and_index_embeddings()
    
    print("\n" + "=" * 60)
    print("🧪 TESTING VECTOR STORE")
    print("=" * 60)
    
    embeddings = np.load("models/embeddings/embeddings.npy")
    test_embedding = embeddings[0]  
    
    print(f"\n Running test query...")
    results = vector_store.query(test_embedding, n_results=3)
    
    print(f"\n Found {len(results['ids'][0])} results:")
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
    print(" VECTOR STORE TEST COMPLETE!")
    print("=" * 60)