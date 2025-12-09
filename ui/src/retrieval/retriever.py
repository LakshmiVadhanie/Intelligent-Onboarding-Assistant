from sentence_transformers import SentenceTransformer
from .vector_store import VectorStore
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineRetriever:
    def __init__(self, 
                 embedding_model: str = "all-mpnet-base-v2",
                 vector_store_dir: str = "models/vector_store",
                 collection_name: str = "gitlab_onboarding"):

        logger.info("Initializing BaselineRetriever...")
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.model_name = embedding_model
        
        logger.info(f"Connecting to vector store: {vector_store_dir}")
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=vector_store_dir
        )
        
        logger.info(" BaselineRetriever initialized successfully!")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:

        logger.info(f"Query: {query}")
        logger.info(f"Retrieving top-{k} documents...")
        
        query_embedding = self.encoder.encode(query, normalize_embeddings=True)
        
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=k
        )
        
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            doc = {
                'rank': i + 1,
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i],  
                'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {}
            }
            retrieved_docs.append(doc)
        
        logger.info(f" Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs
    
    def print_results(self, results: List[Dict], show_full_text: bool = False):
        """
        Pretty print retrieval results
        
        Args:
            results: List of retrieved documents
            show_full_text: Whether to show full document text
        """
        print("\n" + "=" * 80)
        print(f" RETRIEVAL RESULTS ({len(results)} documents)")
        print("=" * 80)
        
        for doc in results:
            print(f"\n🔹 Rank {doc['rank']}")
            print(f"   ID: {doc['id']}")
            print(f"   Similarity: {doc['similarity']:.4f}")
            print(f"   Distance: {doc['distance']:.4f}")
            
            if doc['metadata']:
                print(f"   Title: {doc['metadata'].get('title', 'N/A')}")
                print(f"   Source: {doc['metadata'].get('source', 'N/A')}")
            
            if show_full_text:
                print(f"   Text: {doc['document']}")
            else:
                preview = doc['document'][:200] + "..." if len(doc['document']) > 200 else doc['document']
                print(f"   Preview: {preview}")
            
            print("-" * 80)


# Test the retriever
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" TESTING BASELINE RETRIEVER")
    print("=" * 80)
    
    retriever = BaselineRetriever()
    
    test_queries = [
        "What is GitLab's remote work policy?",
        "How do I submit a pull request?",
        "Tell me about CI/CD pipeline",
        "What are the company values?"
    ]
    
    for query in test_queries:
        print(f"\n\n{'='*80}")
        print(f" QUERY: {query}")
        print('='*80)
        
        results = retriever.retrieve(query, k=3)
        
        retriever.print_results(results, show_full_text=False)
        
        input("\n Press Enter to continue to next query...")
    
    print("\n" + "=" * 80)
    print(" RETRIEVER TEST COMPLETE!")
    print("=" * 80)