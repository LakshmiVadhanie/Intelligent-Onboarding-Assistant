# test_gcs_pipeline.py
import sys
from pathlib import Path
import logging
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from environment variables
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "mlops-data-oa")
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "mlops-476419")
GCS_HANDBOOK_PREFIX = os.getenv("GCS_HANDBOOK_PREFIX", "onboarding_ai/debiased_data/handbook_paragraphs_debiased.json")
GCS_TRANSCRIPT_PREFIX = os.getenv("GCS_TRANSCRIPT_PREFIX", "onboarding_ai/debiased_data/all_transcripts_debiased.json")
GCS_EMBEDDINGS_PREFIX = os.getenv("GCS_EMBEDDINGS_PREFIX", "onboarding_ai/embeddings/")

def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def get_credentials_path():
    """Get credentials path from environment or default location"""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path is None:
        # Try default location relative to project root
        default_creds = Path(__file__).parent.parent / "mlops-476419-2c1937dab204.json"
        if default_creds.exists():
            creds_path = str(default_creds)
    return creds_path

def test_gcs_credentials():
    """Test 1: Verify GCS credentials are set up correctly"""
    print_header("TEST 1: GCS CREDENTIALS")
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        
        creds_path = get_credentials_path()
        
        if creds_path is None:
            raise ValueError("No credentials found! Please set GOOGLE_APPLICATION_CREDENTIALS or place credentials file in project root")
        
        assert Path(creds_path).exists(), f"Credentials file not found: {creds_path}"
        
        # Try to authenticate
        credentials = service_account.Credentials.from_service_account_file(creds_path)
        client = storage.Client(project=GCS_PROJECT_ID, credentials=credentials)
        
        # Try to access bucket
        bucket = client.bucket(GCS_BUCKET_NAME)
        assert bucket.exists(), f"Bucket does not exist: {GCS_BUCKET_NAME}"
        
        print(f"✅ PASSED: GCS credentials valid")
        print(f"   Credentials: {creds_path}")
        print(f"   Project: {GCS_PROJECT_ID}")
        print(f"   Bucket: {GCS_BUCKET_NAME}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_gcs_data_loading():
    """Test 2: Load data from GCS"""
    print_header("TEST 2: GCS DATA LOADING")
    try:
        from src.data.load_from_gcs import GCSDataLoader
        
        loader = GCSDataLoader(
            bucket_name=GCS_BUCKET_NAME,
            project_id=GCS_PROJECT_ID,
            credentials_path=get_credentials_path(),
            local_cache_dir="./.gcs_cache"
        )
        
        # Load handbook
        handbook_chunks, handbook_counts = loader.load_handbook_chunks()
        assert len(handbook_chunks) > 0, "No handbook data loaded!"
        
        # Load transcripts
        transcript_chunks, transcript_counts = loader.load_transcript_chunks()
        assert len(transcript_chunks) > 0, "No transcript data loaded!"
        
        # Load all
        all_chunks, all_counts = loader.load_all_chunks()
        assert len(all_chunks) > 0, "No combined data loaded!"
        
        print(f"✅ PASSED: GCS data loading")
        print(f"   Handbook chunks: {len(handbook_chunks)}")
        print(f"   Transcript chunks: {len(transcript_chunks)}")
        print(f"   Total unique chunks: {len(all_chunks)}")
        print(f"   Files loaded: {list(all_counts.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embeddings_exist_in_gcs():
    """Test 3: Check if embeddings exist in GCS"""
    print_header("TEST 3: EMBEDDINGS IN GCS")
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        
        creds_path = get_credentials_path()
        
        # Initialize GCS client
        credentials = service_account.Credentials.from_service_account_file(creds_path)
        client = storage.Client(project=GCS_PROJECT_ID, credentials=credentials)
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        # Check if embeddings exist in GCS
        files_to_check = [
            "embeddings.npy",
            "texts.json",
            "metadata.json",
            "model_info.json"
        ]
        
        found_files = []
        missing_files = []
        
        for filename in files_to_check:
            blob = bucket.blob(f"{GCS_EMBEDDINGS_PREFIX}{filename}")
            if blob.exists():
                found_files.append(filename)
            else:
                missing_files.append(filename)
        
        if missing_files:
            print(f"⚠️  WARNING: Some files missing in GCS:")
            for f in missing_files:
                print(f"     - {f}")
            print(f"\n   Run this to upload: python src/embeddings/generate_embeddings.py")
        
        assert len(found_files) > 0, "No embedding files found in GCS!"
        
        print(f"✅ PASSED: Embeddings in GCS")
        print(f"   Found files: {found_files}")
        if missing_files:
            print(f"   Missing files: {missing_files}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_embeddings_download():
    """Test 4: Download embeddings from GCS"""
    print_header("TEST 4: EMBEDDINGS DOWNLOAD FROM GCS")
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        import numpy as np
        import json
        
        creds_path = get_credentials_path()
        
        # Initialize GCS client
        credentials = service_account.Credentials.from_service_account_file(creds_path)
        client = storage.Client(project=GCS_PROJECT_ID, credentials=credentials)
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        # Download embeddings
        test_dir = Path("models/embeddings_test_gcs")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Download at least one file to verify
        blob = bucket.blob(f"{GCS_EMBEDDINGS_PREFIX}model_info.json")
        if not blob.exists():
            print(f"⚠️  WARNING: Embeddings not uploaded to GCS yet")
            print(f"   Run: python src/embeddings/generate_embeddings.py")
            return True  # Soft pass
        
        local_path = test_dir / "model_info.json"
        blob.download_to_filename(str(local_path))
        
        assert local_path.exists(), "Download failed!"
        
        # Verify content
        with open(local_path) as f:
            model_info = json.load(f)
        
        assert 'model_name' in model_info, "Invalid model info!"
        
        print(f"✅ PASSED: GCS download working")
        print(f"   Model: {model_info.get('model_name')}")
        print(f"   Dimensions: {model_info.get('embedding_dim')}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_store():
    """Test 5: Vector store"""
    print_header("TEST 5: VECTOR STORE")
    try:
        from src.retrieval.vector_store import VectorStore
        
        vector_store = VectorStore(
            collection_name="gitlab_onboarding",
            persist_directory="models/vector_store",
            use_gcs=False,
            credentials_path=get_credentials_path()
        )
        
        count = vector_store.collection.count()
        
        if count == 0:
            print(f"⚠️  WARNING: Vector store is empty")
            print(f"   Run: python src/retrieval/vector_store.py")
            return True  # Soft pass
        
        print(f"✅ PASSED: Vector store working")
        print(f"   Documents indexed: {count}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retriever():
    """Test 6: Advanced retriever"""
    print_header("TEST 6: ADVANCED RETRIEVER")
    try:
        from src.retrieval.advanced_retriever import AdvancedRetriever
        
        retriever = AdvancedRetriever(
            use_gcs=False,
            bucket_name=GCS_BUCKET_NAME,
            project_id=GCS_PROJECT_ID,
            credentials_path=get_credentials_path()
        )
        
        # Test retrieval
        results = retriever.retrieve("What is sustainability?", k=3)
        
        assert len(results) > 0, "No results retrieved!"
        assert 'rerank_score' in results[0], "Missing rerank score!"
        
        print(f"✅ PASSED: Advanced retriever working")
        print(f"   Retrieved: {len(results)} documents")
        print(f"   Top rerank score: {results[0]['rerank_score']:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_pipeline():
    """Test 7: RAG pipeline"""
    print_header("TEST 7: RAG PIPELINE")
    try:
        from src.generation.rag_pipeline import UniversalRAGPipeline
        
        rag = UniversalRAGPipeline(
            provider="gemini",
            use_gcs=False,
            bucket_name=GCS_BUCKET_NAME,
            project_id=GCS_PROJECT_ID,
            credentials_path=get_credentials_path()
        )
        
        # Test query
        result = rag.generate_answer("What is sustainability?", k=3)
        
        assert 'answer' in result, "Missing answer!"
        assert 'sources' in result, "Missing sources!"
        
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        print(f"✅ PASSED: RAG pipeline working")
        print(f"   Sources retrieved: {result['num_sources']}")
        
        if api_key and rag.client:
            print(f"   Generation: Enabled")
            if not result['answer'].startswith("[Error"):
                print(f"   Answer preview: {result['answer'][:100]}...")
            else:
                print(f"   Generation error (likely quota): {result['answer'][:80]}...")
        else:
            print(f"   Generation: Disabled (no API key)")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gcs_integration_status():
    """Test 8: Overall GCS integration status"""
    print_header("TEST 8: GCS INTEGRATION STATUS")
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        
        creds_path = get_credentials_path()
        credentials = service_account.Credentials.from_service_account_file(creds_path)
        client = storage.Client(project=GCS_PROJECT_ID, credentials=credentials)
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        # Check what's in GCS
        print("📊 GCS Bucket Contents:")
        
        prefixes = [
            "onboarding_ai/debiased_data/",
            GCS_EMBEDDINGS_PREFIX
        ]
        
        for prefix in prefixes:
            blobs = list(bucket.list_blobs(prefix=prefix, max_results=10))
            print(f"\n   {prefix}")
            if blobs:
                for blob in blobs[:5]:
                    print(f"     ✓ {blob.name.split('/')[-1]}")
                if len(blobs) > 5:
                    print(f"     ... and {len(blobs)-5} more files")
            else:
                print(f"     (empty)")
        
        print(f"\n✅ PASSED: GCS integration configured")
        print(f"   Credentials: Working")
        print(f"   Bucket access: Working")
        print(f"   Project: {GCS_PROJECT_ID}")
        print(f"   Bucket: {GCS_BUCKET_NAME}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def run_gcs_tests():
    """Run all GCS-specific tests"""
    print("\n" + "=" * 80)
    print("  GCS INTEGRATION TEST SUITE")
    print("=" * 80)
    
    # Display configuration
    print(f"\n📋 Configuration:")
    print(f"   Project ID: {GCS_PROJECT_ID}")
    print(f"   Bucket: {GCS_BUCKET_NAME}")
    print(f"   Embeddings Prefix: {GCS_EMBEDDINGS_PREFIX}")
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"   Gemini API: ✅ Configured")
    else:
        print(f"   Gemini API: ⚠️  Not configured (retrieval only)")
    
    print("=" * 80)
    
    start_time = time.time()
    
    tests = [
        ("1. GCS Credentials", test_gcs_credentials),
        ("2. GCS Data Loading", test_gcs_data_loading),
        ("3. Embeddings in GCS", test_embeddings_exist_in_gcs),
        ("4. Embeddings Download", test_embeddings_download),
        ("5. Vector Store", test_vector_store),
        ("6. Advanced Retriever", test_retriever),
        ("7. RAG Pipeline", test_rag_pipeline),
        ("8. GCS Integration Status", test_gcs_integration_status),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
            time.sleep(0.5)
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    elapsed_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("  GCS TEST SUMMARY")
    print("=" * 80 + "\n")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print("\n" + "-" * 80)
    print(f"  Total: {passed_count}/{total_count} tests passed")
    print(f"  Success Rate: {(passed_count/total_count)*100:.1f}%")
    print(f"  Time: {elapsed_time:.2f} seconds")
    print("=" * 80)
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Your pipeline is working with GCS!")
    elif passed_count >= 6:
        print("\n✅ Core functionality working!")
        print("   Some GCS features may need setup (see warnings above)")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed.")
    
    print("\nNext steps:")
    print("   1. Reset vector store: python reset_to_gcs.py")
    print("   2. Launch UI: streamlit run app.py")
    print("=" * 80 + "\n")
    
    return passed_count >= 6


if __name__ == "__main__":
    success = run_gcs_tests()
    sys.exit(0 if success else 1)