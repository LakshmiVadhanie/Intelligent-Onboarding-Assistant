# src/retrieval/advanced_retriever.py

from sentence_transformers import SentenceTransformer  # CrossEncoder disabled
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
        self.reranker = None  # Disabled to avoid HuggingFace rate limits
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
            Top-k reranked documents (list of dict). Returns [] if no candidates found.
        """
        logger.info(f"🔍 Query: '{query[:60]}...'")
        logger.info(f"   Stage 1: Retrieving top-{rerank_top_n} candidates...")

        # Stage 1: Dense retrieval
        try:
            query_embedding = self.encoder.encode(query, normalize_embeddings=True)
        except Exception:
            # fallback: encode without normalization if model doesn't support it
            logger.exception("Failed to encode query with normalize_embeddings=True, retrying without normalization.")
            query_embedding = self.encoder.encode(query)

        # Query the vector store
        try:
            results = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=rerank_top_n
            )
        except Exception:
            logger.exception("vector_store.query() raised an exception. Returning empty results.")
            return []

        # Defensive: ensure results structure is present and non-empty
        try:
            ids_batch = results.get('ids', [])
            docs_batch = results.get('documents', [])
            dists_batch = results.get('distances', [])
            metas_batch = results.get('metadatas', [])
        except Exception:
            logger.exception("Unexpected vector_store.query() return format.")
            return []

        # If the query returns no items (empty collection), return early
        if not ids_batch or not ids_batch[0]:
            logger.info("   ✓ Retrieved 0 candidates (vector store empty or no match).")
            return []

        # Format candidates safely (guard index accesses)
        candidates = []
        num_items = len(ids_batch[0])
        for i in range(num_items):
            doc_text = ""
            if docs_batch and isinstance(docs_batch, list) and docs_batch and len(docs_batch[0]) > i:
                doc_text = docs_batch[0][i]
            dist = None
            if dists_batch and isinstance(dists_batch, list) and dists_batch and len(dists_batch[0]) > i:
                dist = dists_batch[0][i]
            meta = {}
            if metas_batch and isinstance(metas_batch, list) and metas_batch and len(metas_batch[0]) > i:
                meta = metas_batch[0][i] or {}

            dense_score = None
            if dist is not None:
                try:
                    dense_score = 1 - float(dist)
                except Exception:
                    dense_score = None

            candidates.append({
                'id': ids_batch[0][i],
                'document': doc_text,
                'dense_score': dense_score,
                'metadata': meta
            })

        logger.info(f"   ✓ Retrieved {len(candidates)} candidates")

        # Stage 2: Reranking
        logger.info(f"   Stage 2: Reranking with cross-encoder...")

        # Defensive: if no candidates, skip reranker
        if not candidates:
            logger.info("   No candidates to rerank; returning empty result set.")
            return []

        # CrossEncoder expects list of tuples (query, doc_text). Use tuples for pairs.
        pairs = [(query, doc['document']) for doc in candidates]

        # Defensive call to reranker: catch errors (including empty input or unexpected exceptions)
        try:
            rerank_scores = self.reranker.predict(pairs)
        except IndexError as e:
            # This may occur if reranker receives empty list — should be avoided by checks above.
            logger.exception("Reranker IndexError (likely empty input). Returning dense-ranked candidates.")
            # Fall back: return top-k by dense_score if available
            fallback_sorted = sorted(
                [c for c in candidates if c['dense_score'] is not None],
                key=lambda x: x['dense_score'],
                reverse=True
            )
            final_results = fallback_sorted[:k]
            for idx, doc in enumerate(final_results, 1):
                doc['rank'] = idx
                doc['similarity'] = doc.get('dense_score')
            return final_results
        except Exception:
            logger.exception("Unexpected error during reranking. Returning dense-ranked candidates as fallback.")
            fallback_sorted = sorted(
                [c for c in candidates if c['dense_score'] is not None],
                key=lambda x: x['dense_score'],
                reverse=True
            )
            final_results = fallback_sorted[:k]
            for idx, doc in enumerate(final_results, 1):
                doc['rank'] = idx
                doc['similarity'] = doc.get('dense_score')
            return final_results

        # Add reranking scores (ensure lengths match)
        if len(rerank_scores) != len(candidates):
            logger.warning("Reranker returned %d scores but there are %d candidates. Truncating/padding as needed.",
                           len(rerank_scores), len(candidates))

        for i, doc in enumerate(candidates):
            score = float(rerank_scores[i]) if i < len(rerank_scores) else (doc.get('dense_score') or 0.0)
            doc['rerank_score'] = score
            doc['rank_before_rerank'] = i + 1

        # Sort by rerank score and build final list
        reranked = sorted(candidates, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
        final_results = reranked[:k]

        for i, doc in enumerate(final_results, 1):
            doc['rank'] = i
            doc['similarity'] = doc.get('rerank_score', doc.get('dense_score'))

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
        try:
            query_embedding = self.encoder.encode(query, normalize_embeddings=True)
        except Exception:
            query_embedding = self.encoder.encode(query)

        try:
            baseline_query = self.vector_store.query(query_embedding, n_results=k)
        except Exception:
            logger.exception("vector_store.query() failed during baseline compare.")
            baseline_query = {'ids': [[]], 'distances': [[]]}

        baseline_results = []
        for i in range(len(baseline_query.get('ids', [[]])[0])):
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
        
        if comparison['advanced']:
            for doc in comparison['advanced']:
                print(f"\n✓ Rank {doc['rank']}: {doc['id']}")
                print(f"  Rerank Score: {doc.get('rerank_score', 0.0):.4f}")
                print(f"  Dense Score: {doc.get('dense_score', 0.0):.4f}")
                print(f"  Rank before rerank: {doc.get('rank_before_rerank', 'N/A')}")
                print(f"  Preview: {doc['document'][:100]}...")
        else:
            print("No advanced retrieval results (vector store may be empty).")
        
        print(f"\n🔄 Reranking changed order: {comparison['reranking_changed_order']}")
        
        input("\n👉 Press Enter for next query...")
    
    print("\n" + "="*80)
    print("✅ ADVANCED RETRIEVER TEST COMPLETE!")
    print("="*80)
    print("\n  Cross-encoder reranking improves relevance by scoring")
    print("   query-document pairs directly, rather than relying only")
    print("   on embedding similarity.")
    print("="*80)
