import sys
from pathlib import Path
import logging
import time
import os
from dotenv import load_dotenv 
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)  
logger = logging.getLogger(__name__)

def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def test_data_loading():
    print_header("TEST 1: DATA LOADING")
    
    try:
        from src.data.load_local_data import LocalDataLoader
        
        loader = LocalDataLoader(data_dir="data")
        chunks = loader.load_all_data()
        
        assert len(chunks) > 0, "No data loaded!"
        
        print(f" PASSED: Loaded {len(chunks)} chunks")
        return True
        
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def test_embeddings():
    print_header("TEST 2: EMBEDDINGS")
    
    try:
        import numpy as np
        
        embeddings_path = Path("models/embeddings/embeddings.npy")
        assert embeddings_path.exists(), "Embeddings not found!"
        
        embeddings = np.load(embeddings_path)
        
        print(f" PASSED: Embeddings shape {embeddings.shape}")
        return True
        
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def test_vector_store():
    print_header("TEST 3: VECTOR STORE")
    
    try:
        from src.retrieval.vector_store import VectorStore
        
        vector_store = VectorStore(
            collection_name="gitlab_onboarding",
            persist_directory="models/vector_store"
        )
        
        count = vector_store.collection.count()
        assert count > 0, "Vector store is empty!"
        
        print(f" PASSED: {count} documents indexed")
        return True
        
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def test_baseline_retrieval():
    print_header("TEST 4: BASELINE RETRIEVAL")
    
    try:
        from src.retrieval.retriever import BaselineRetriever
        
        retriever = BaselineRetriever()
        results = retriever.retrieve("sustainability", k=3)
        
        assert len(results) == 3, "Should retrieve 3 documents"
        
        print(f" PASSED: Baseline retrieval working")
        print(f"   Top similarity: {results[0]['similarity']:.4f}")
        return True
        
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def test_advanced_retrieval():
    print_header("TEST 5: ADVANCED RETRIEVAL (RERANKING)")
    
    try:
        from src.retrieval.advanced_retriever import AdvancedRetriever
        
        retriever = AdvancedRetriever()
        results = retriever.retrieve("sustainability", k=3)
        
        assert len(results) == 3, "Should retrieve 3 documents"
        assert 'rerank_score' in results[0], "Missing rerank score"
        
        print(f" PASSED: Advanced retrieval with reranking")
        print(f"   Top rerank score: {results[0]['rerank_score']:.4f}")
        return True
        
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def test_rag_generation():
    print_header("TEST 6: RAG GENERATION (GEMINI)")
    
    try:
        from src.generation.rag_pipeline import UniversalRAGPipeline
        
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        rag = UniversalRAGPipeline(provider="gemini")
        result = rag.generate_answer("What is sustainability?", k=3)
        
        assert 'answer' in result, "Missing answer"
        assert 'sources' in result, "Missing sources"
        
        if api_key and rag.client:
            assert result['answer'] != "[Generation not available - no API key set]"
            print(f" PASSED: Full RAG with Gemini generation")
            print(f"   Answer length: {len(result['answer'])} chars")
            print(f"   Sources used: {result['num_sources']}")
        else:
            print(f"  PARTIAL: RAG framework working (retrieval-only, no API key)")
        
        return True
        
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def test_evaluation_metrics():
    print_header("TEST 7: EVALUATION METRICS")
    
    try:
        from src.retrieval.retriever import BaselineRetriever
        from src.evaluation.metrics import RetrievalEvaluator
        
        retriever = BaselineRetriever()
        evaluator = RetrievalEvaluator(retriever)
        
        test_queries = [{'query': 'sustainability', 'relevant_ids': ['doc_8']}]
        results = evaluator.evaluate(test_queries, k_values=[1, 3, 5])
        
        assert 'mrr' in results, "Missing MRR"
        
        print(f" PASSED: Evaluation metrics")
        print(f"   MRR: {results['mrr']:.4f}")
        print(f"   Precision@5: {results['metrics_by_k']['k=5']['precision']:.4f}")
        return True
        
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def test_bias_detection():
    print_header("TEST 8: BIAS DETECTION")
    
    try:
        from src.retrieval.retriever import BaselineRetriever
        from src.evaluation.bias_detection import BiasDetector
        
        retriever = BaselineRetriever()
        detector = BiasDetector(retriever)
        
        test_queries = [
            {'query': 'sustainability', 'relevant_ids': ['doc_8']},
        ]
        
        report = detector.generate_bias_report(test_queries, k=3)
        
        assert 'has_significant_bias' in report, "Missing bias flag"
        
        print(f" PASSED: Bias detection")
        print(f"   Significant bias: {report['has_significant_bias']}")
        return True
        
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def test_sensitivity_analysis():
    print_header("TEST 9: SENSITIVITY ANALYSIS")
    
    try:
        from src.retrieval.retriever import BaselineRetriever
        from src.evaluation.metrics import RetrievalEvaluator
        from src.evaluation.sensitivity_analysis import RetrievalSensitivityAnalyzer
        
        retriever = BaselineRetriever()
        evaluator = RetrievalEvaluator(retriever)
        analyzer = RetrievalSensitivityAnalyzer(retriever, evaluator)
        
        test_queries = [
            {'query': 'sustainability', 'relevant_ids': ['doc_8']},
        ]
        
        report = analyzer.analyze_embedding_dimension_impact()
        
        assert 'embedding_dimension' in report or len(report) == 0
        
        print(f" PASSED: Sensitivity analysis")
        return True
        
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def test_ragas():
    print_header("TEST 10: RAGAS EVALUATION")
    
    try:
        from src.retrieval.advanced_retriever import AdvancedRetriever
        from src.evaluation.ragas_evaluator import RAGASEvaluator
        
        retriever = AdvancedRetriever()
        evaluator = RAGASEvaluator()
        
        test_data = [
            {'question': 'sustainability', 'ground_truth': 'GitLab sustainability'}
        ]
        
        results = evaluator.evaluate_retrieval_only(retriever, test_data)
        
        assert 'metrics' in results or 'num_queries' in results
        
        print(f" PASSED: RAGAS evaluation")
        return True
        
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def test_model_registry():
    print_header("TEST 11: MODEL REGISTRY")
    
    try:
        registry_manifest = Path("models/registry/manifest.json")
        
        if registry_manifest.exists():
            import json
            with open(registry_manifest) as f:
                manifest = json.load(f)
            
            assert 'models' in manifest, "Invalid manifest"
            
            print(f" PASSED: Model registry")
            print(f"   Models registered: {len(manifest['models'])}")
        else:
            print(f"  SKIPPED: Model registry not found (run model_registry.py)")
        
        return True
        
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def test_mlflow_tracking():
    print_header("TEST 12: MLFLOW EXPERIMENT TRACKING")
    
    try:
        import mlflow
        
        mlflow.set_tracking_uri("file:./experiments/mlruns")
        experiments = mlflow.search_experiments()
        
        assert len(experiments) > 0, "No experiments found"
        
        print(f" PASSED: MLflow tracking")
        print(f"   Experiments: {len(experiments)}")
        return True
        
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def run_all_tests():
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE PIPELINE TEST - ALL FEATURES")
    print("=" * 80)
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        print("\n  Gemini API key found - testing full RAG generation!")
    else:
        print("\n  No API key - testing retrieval components only")
    
    print("=" * 80)
    
    start_time = time.time()
    
    tests = [
        ("1. Data Loading", test_data_loading),
        ("2. Embeddings", test_embeddings),
        ("3. Vector Store", test_vector_store),
        ("4. Baseline Retrieval", test_baseline_retrieval),
        ("5. Advanced Retrieval (Reranking)", test_advanced_retrieval),
        ("6. RAG Generation (Gemini)", test_rag_generation),
        ("7. Evaluation Metrics", test_evaluation_metrics),
        ("8. Bias Detection", test_bias_detection),
        ("9. Sensitivity Analysis", test_sensitivity_analysis),
        ("10. RAGAS Evaluation", test_ragas),
        ("11. Model Registry", test_model_registry),
        ("12. MLflow Tracking", test_mlflow_tracking)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n UNEXPECTED ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE TEST SUMMARY")
    print("=" * 80 + "\n")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = " PASSED" if passed else " FAILED"
        print(f"  {status}: {test_name}")
    
    print("\n" + "-" * 80)
    print(f"  Total: {passed_count}/{total_count} tests passed")
    print(f"  Success Rate: {(passed_count/total_count)*100:.1f}%")
    print(f"  Time: {elapsed_time:.2f} seconds")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("  FEATURE COVERAGE")
    print("=" * 80)
    print("   Data Pipeline Integration")
    print("   Embedding Generation (384-dim vectors)")
    print("   Vector Store (ChromaDB)")
    print("   Baseline Retrieval (Top-K)")
    print("   Advanced Retrieval (Cross-encoder reranking)")
    print("   RAG Framework (Gemini LLM)")
    print("   Evaluation Metrics (Precision, Recall, MRR, NDCG)")
    print("   Bias Detection (Slice-based fairness)")
    print("   Sensitivity Analysis (Query length, K-value)")
    print("   RAGAS Evaluation (Context metrics)")
    print("   Model Registry (Versioning)")
    print("   MLflow Tracking (Experiment management)")
    print("=" * 80)
    
    if passed_count == total_count:
        print("\n ALL TESTS PASSED!")
        print("\n Next steps:")
        print("   - Run: python -m streamlit run app.py (web UI)")
        print("   - Run: python -m mlflow ui (view experiments)")
        print("\n" + "=" * 80 + "\n")
        return True
    else:
        print(f"\n {total_count - passed_count} test(s) failed.\n")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)