from sentence_transformers import SentenceTransformer, CrossEncoder
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from vector_store import VectorStore

import numpy as np
from typing import List, Dict
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRetriever:
    """Advanced retriever with cross-encoder reranking and GCS support"""
    
    def __init__(self, 
                 embedding_model: str = "all-mpnet-base-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 vector_store_dir: str = "models/vector_store",
                 collection_name: str = "gitlab_onboarding",
                 use_gcs: bool = True,
                 bucket_name: str = "mlops-data-oa",
                 project_id: str = "mlops-476419",
                 credentials_path: str = None):
        """
        Initialize advanced retriever with GCS support
        
        Args:
            embedding_model: Dense retrieval model
            reranker_model: Cross-encoder for reranking
            vector_store_dir: Vector store directory
            collection_name: ChromaDB collection name
            use_gcs: Whether vector store uses GCS
            bucket_name: GCS bucket name
            project_id: GCP project ID
            credentials_path: Path to service account credentials
        """
        logger.info("=" * 80)
        logger.info("Initializing AdvancedRetriever with reranking...")
        logger.info("=" * 80)
        
        # Resolve credentials path
        if credentials_path is None:
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_path is None:
                default_creds = Path(__file__).resolve().parents[3] / "mlops-476419-2c1937dab204.json"
                if default_creds.exists():
                    credentials_path = str(default_creds)
        
        logger.info(f"\n📚 Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_model_name = embedding_model
        logger.info(f"✓ Loaded embedding model")
        
        logger.info(f"\n🔄 Loading reranker model: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)
        self.reranker_model_name = reranker_model
        logger.info(f"✓ Loaded reranker model")
        
        logger.info(f"\n📊 Connecting to vector store: {vector_store_dir}")
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=vector_store_dir,
            use_gcs=use_gcs,
            bucket_name=bucket_name,
            project_id=project_id,
            credentials_path=credentials_path
        )
        logger.info(f"✓ Connected to vector store")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ADVANCED RETRIEVER INITIALIZED!")
        logger.info("=" * 80 + "\n")
    
    def retrieve(self, 
                query: str, 
                k: int = 5, 
                rerank_top_n: int = 20) -> List[Dict]:
        """
        Retrieve with two-stage approach: dense retrieval + reranking
        
        Args:
            query: User query
            k: Final number of documents to return
            rerank_top_n: Number of candidates to retrieve before reranking
            
        Returns:
            Top-k reranked documents
        """
        logger.info(f"🔍 Query: '{query[:60]}...'")
        logger.info(f"   Stage 1: Retrieving top-{rerank_top_n} candidates...")
        
        # Stage 1: Dense retrieval
        query_embedding = self.encoder.encode(query, normalize_embeddings=True)
        
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=rerank_top_n
        )
        
        # Format candidates
        candidates = []
        for i in range(len(results['ids'][0])):
            candidates.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'dense_score': 1 - results['distances'][0][i],  
                'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {}
            })
        
        logger.info(f"   ✓ Retrieved {len(candidates)} candidates")
        
        # Stage 2: Reranking
        logger.info(f"   Stage 2: Reranking with cross-encoder...")
        
        pairs = [[query, doc['document']] for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        # Add reranking scores
        for i, doc in enumerate(candidates):
            doc['rerank_score'] = float(rerank_scores[i])
            doc['rank_before_rerank'] = i + 1
        
        # Sort by rerank score
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        # Get top-k
        final_results = reranked[:k]
        
        # Add final ranks
        for i, doc in enumerate(final_results, 1):
            doc['rank'] = i
            doc['similarity'] = doc['rerank_score'] 
        
        logger.info(f"   ✓ Reranked and returning top-{k} documents")
        
        return final_results
    
    def compare_with_baseline(self, query: str, k: int = 5) -> Dict:
        """
        Compare advanced retrieval vs baseline
        
        Args:
            query: Test query
            k: Number of results
            
        Returns:
            Comparison results
        """
        # Get advanced results
        advanced_results = self.retrieve(query, k=k)
        
        # Get baseline results (dense only)
        query_embedding = self.encoder.encode(query, normalize_embeddings=True)
        baseline_query = self.vector_store.query(query_embedding, n_results=k)
        
        baseline_results = []
        for i in range(len(baseline_query['ids'][0])):
            baseline_results.append({
                'id': baseline_query['ids'][0][i],
                'rank': i + 1,
                'score': 1 - baseline_query['distances'][0][i]
            })
        
        return {
            'query': query,
            'advanced': advanced_results,
            'baseline': baseline_results,
            'reranking_changed_order': [
                doc['id'] for doc in advanced_results
            ] != [doc['id'] for doc in baseline_results]
        }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 TESTING ADVANCED RETRIEVER WITH RERANKING")
    print("="*80)
    
    # Get credentials path
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path is None:
        default_creds = Path(__file__).resolve().parents[3] / "mlops-476419-2c1937dab204.json"
        if default_creds.exists():
            creds_path = str(default_creds)
    
    # Initialize retriever with GCS support
    retriever = AdvancedRetriever(
        use_gcs=True,
        bucket_name="mlops-data-oa",
        project_id="mlops-476419",
        credentials_path=creds_path
    )
    
    test_queries = [
        "What is GitLab's approach to sustainability?",
        "How does risk management work?",
        "Tell me about CI/CD processes"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"📝 QUERY: {query}")
        print('='*80)
        
        comparison = retriever.compare_with_baseline(query, k=3)
        
        print(f"\n🎯 ADVANCED RETRIEVAL RESULTS (with reranking):")
        print("-"*80)
        
        for doc in comparison['advanced']:
            print(f"\n✓ Rank {doc['rank']}: {doc['id']}")
            print(f"  Rerank Score: {doc['rerank_score']:.4f}")
            print(f"  Dense Score: {doc['dense_score']:.4f}")
            print(f"  Rank before rerank: {doc['rank_before_rerank']}")
            print(f"  Preview: {doc['document'][:100]}...")
        
        print(f"\n🔄 Reranking changed order: {comparison['reranking_changed_order']}")
        
        input("\n👉 Press Enter for next query...")
    
    print("\n" + "="*80)
    print("✅ ADVANCED RETRIEVER TEST COMPLETE!")
    print("="*80)
    print("\n  Cross-encoder reranking improves relevance by scoring")
    print("   query-document pairs directly, rather than relying only")
    print("   on embedding similarity.")
    print("="*80)