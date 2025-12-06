from sentence_transformers import SentenceTransformer, CrossEncoder
from .vector_store import VectorStore
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRetriever:
    """Advanced retriever with cross-encoder reranking"""
    
    def __init__(self, 
                 embedding_model: str = "all-mpnet-base-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 vector_store_dir: str = "models/vector_store",
                 collection_name: str = "gitlab_onboarding"):
        """
        Initialize advanced retriever
        
        Args:
            embedding_model: Dense retrieval model
            reranker_model: Cross-encoder for reranking
            vector_store_dir: Vector store directory
            collection_name: ChromaDB collection name
        """
        logger.info("Initializing AdvancedRetriever with reranking...")
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_model_name = embedding_model
        
        logger.info(f"Loading reranker model: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)
        self.reranker_model_name = reranker_model
        
        logger.info(f"Connecting to vector store: {vector_store_dir}")
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=vector_store_dir
        )
        
        logger.info(" AdvancedRetriever initialized with reranking!")
    
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
        logger.info(f"Advanced retrieval for: {query[:60]}...")
        logger.info(f"Retrieving top-{rerank_top_n} candidates for reranking...")
        
        query_embedding = self.encoder.encode(query, normalize_embeddings=True)
        
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=rerank_top_n
        )
        
        candidates = []
        for i in range(len(results['ids'][0])):
            candidates.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'dense_score': 1 - results['distances'][0][i],  
                'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {}
            })
        
        logger.info(f"✅ Retrieved {len(candidates)} candidates")
        
        logger.info(f"Reranking top-{rerank_top_n} candidates...")
        
        pairs = [[query, doc['document']] for doc in candidates]
        
        rerank_scores = self.reranker.predict(pairs)
        
        for i, doc in enumerate(candidates):
            doc['rerank_score'] = float(rerank_scores[i])
            doc['rank_before_rerank'] = i + 1
        
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        final_results = reranked[:k]
        
        for i, doc in enumerate(final_results, 1):
            doc['rank'] = i
            doc['similarity'] = doc['rerank_score'] 
        
        logger.info(f"✅ Reranked and returning top-{k} documents")
        
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
        advanced_results = self.retrieve(query, k=k)
        
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
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    print("\n" + "="*80)
    print(" TESTING ADVANCED RETRIEVER WITH RERANKING")
    print("="*80)
    
    retriever = AdvancedRetriever()
    
    test_queries = [
        "What is GitLab's approach to sustainability?",
        "How does risk management work?",
        "Tell me about CI/CD processes"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f" QUERY: {query}")
        print('='*80)
        
        comparison = retriever.compare_with_baseline(query, k=3)
        
        print(f"\n ADVANCED RETRIEVAL RESULTS (with reranking):")
        print("-"*80)
        
        for doc in comparison['advanced']:
            print(f"\nRank {doc['rank']}: {doc['id']}")
            print(f"  Rerank Score: {doc['rerank_score']:.4f}")
            print(f"  Dense Score: {doc['dense_score']:.4f}")
            print(f"  Rank before rerank: {doc['rank_before_rerank']}")
            print(f"  Preview: {doc['document'][:100]}...")
        
        print(f"\n Reranking changed order: {comparison['reranking_changed_order']}")
        
        input("\n  Press Enter for next query...")
    
    print("\n" + "="*80)
    print(" ADVANCED RETRIEVER TEST COMPLETE!")
    print("="*80)
    print("\n  Cross-encoder reranking improves relevance by scoring")
    print("   query-document pairs directly, rather than relying only")
    print("   on embedding similarity.")
    print("="*80)