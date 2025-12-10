from groq import Groq
from groq import Groq
# from openai import OpenAI
import os
from typing import List, Dict, Optional
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval.advanced_retriever import AdvancedRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalRAGPipeline:
    
    def __init__(self, 
                 provider: str = "groq",
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 0.3,
                 use_gcs: bool = True,
                 bucket_name: str = "mlops-data-oa",
                 project_id: str = "mlops-476419",
                 credentials_path: Optional[str] = None):
        """
        Initialize Universal RAG Pipeline with GCS support
        
        Args:
            provider: LLM provider ("gemini", "openai", or "groq")
            api_key: API key for LLM
            model: Model name
            temperature: Generation temperature
            use_gcs: Whether to use GCS for data
            bucket_name: GCS bucket name
            project_id: GCP project ID
            credentials_path: Path to service account credentials
        """
        logger.info("=" * 80)
        logger.info(f"Initializing Universal RAG Pipeline with {provider.upper()}...")
        logger.info("=" * 80)
        
        # Resolve credentials path
        if credentials_path is None:
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_path is None:
                default_creds = Path(__file__).resolve().parents[3] / "mlops-476419-2c1937dab204.json"
                if default_creds.exists():
                    credentials_path = str(default_creds)
        
        logger.info("\n📊 Setting up advanced retriever with reranking...")
        self.retriever = AdvancedRetriever(
            use_gcs=use_gcs,
            bucket_name=bucket_name,
            project_id=project_id,
            credentials_path=credentials_path
        )
        
        self.provider = provider.lower()
        self.temperature = temperature
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.client = None
        self.model_name = model
        
        if self.provider == "gemini":
            self._init_gemini(api_key, model)
        elif self.provider == "openai":
            self._init_openai(api_key, model)
        elif self.provider == "groq":
            self._init_groq(api_key, model)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'gemini', 'openai', or 'groq'")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ UNIVERSAL RAG PIPELINE INITIALIZED!")
        logger.info("=" * 80 + "\n")
    
    def _init_gemini(self, api_key: Optional[str], model: Optional[str]):
        """Initialize Gemini"""
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if api_key is None:
            logger.warning("⚠️ No Google API key found!")
            logger.warning("   Get free key: https://aistudio.google.com/app/apikey")
            logger.warning("   Set: $env:GROQ_API_KEY='your-key'")
            logger.warning("   Pipeline will only retrieve, not generate")
            return
        
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model or "gemini-2.0-flash"
        self.client = genai.GenerativeModel(self.model_name)
        logger.info(f"✓ Gemini initialized: {self.model_name} (FREE!)")
    
    def _init_openai(self, api_key: Optional[str], model: Optional[str]):
        """Initialize OpenAI"""
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key is None:
            logger.warning("⚠️ No OpenAI API key found!")
            logger.warning("   Set: $env:OPENAI_API_KEY='your-key'")
            logger.warning("   Pipeline will only retrieve, not generate")
            return
        
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model or "gpt-3.5-turbo"
        logger.info(f"✓ OpenAI initialized: {self.model_name}")
    
    def _init_groq(self, api_key: Optional[str], model: Optional[str]):
        """Initialize Groq"""
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        
        if api_key is None:
            logger.warning("⚠️ No Groq API key found!")
            logger.warning("   Get free key: https://console.groq.com/keys")
            logger.warning("   Add to .env: GROQ_API_KEY='your-key'")
            logger.warning("   Pipeline will only retrieve, not generate")
            return
        
        self.client = Groq(api_key=api_key)
        self.model_name = model or "llama-3.3-70b-versatile"
        logger.info(f"✓ Groq initialized: {self.model_name} (FREE & FAST!)")
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents"""
        return self.retriever.retrieve(query, k=k)
    
    def build_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Build prompt with retrieved context - optimized for token limits"""
        context_parts = []
        MAX_CHARS_PER_DOC = 500  # ~125 tokens per document (safe for 3 docs)
        
        for i, doc in enumerate(retrieved_docs, 1):
            title = doc['metadata'].get('title', 'Untitled')
            text = doc['document']
            
            # Truncate to fit Groq's limits (critical for large chunks)
            if len(text) > MAX_CHARS_PER_DOC:
                text = text[:MAX_CHARS_PER_DOC] + "..."
                logger.debug(f"Truncated doc {i} from {len(doc['document'])} to {MAX_CHARS_PER_DOC} chars")
            
            context_parts.append(f"[Source {i}: {title}]\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # Concise prompt to save tokens
        prompt = f"""Answer based on GitLab handbook context.

Be concise and cite sources (e.g., "According to Source 1...").

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_answer(self, query: str, k: int = 5) -> Dict:
        """Generate answer using RAG"""
        logger.info("=" * 80)
        logger.info(f"🔍 Processing query: {query}")
        logger.info("=" * 80)
        
        # Retrieve documents
        retrieved_docs = self.retrieve_context(query, k=k)
        logger.info(f"✓ Retrieved {len(retrieved_docs)} documents with reranking")
        
        # If no client, return retrieval only
        if self.client is None:
            logger.warning("⚠️ No LLM client - returning retrieval only")
            return {
                'query': query,
                'answer': "[Generation not available - no API key set]",
                'sources': retrieved_docs,
                'num_sources': len(retrieved_docs),
                'provider': self.provider,
                'model': None
            }
        
        # Build prompt
        prompt = self.build_prompt(query, retrieved_docs)
        logger.info("✓ Built prompt with context")
        
        # Generate answer
        logger.info(f"🤖 Generating answer using {self.provider.upper()}...")
        
        try:
            if self.provider == "gemini":
                answer = self._generate_gemini(prompt)
            elif self.provider == "openai":
                answer = self._generate_openai(prompt)
            elif self.provider == "groq":
                answer = self._generate_groq(prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            logger.info("✓ Answer generated successfully")
            
        except Exception as e:
            logger.error(f"✗ Error generating answer: {e}")
            answer = f"[Error: {str(e)}]"
        
        # Format response
        result = {
            'query': query,
            'answer': answer,
            'sources': retrieved_docs,
            'num_sources': len(retrieved_docs),
            'provider': self.provider,
            'model': self.model_name
        }
        
        return result
    
    def _generate_gemini(self, prompt: str) -> str:
        """Generate with Gemini"""
        response = self.groq_client.chat.completions.create(model="mixtral-8x7b-32768", messages=[{"role": "user", "content": 
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=500,
            }])
        )
        return response.choices[0].message.content
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate with OpenAI"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful GitLab onboarding assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=500
        )
        return response.choices[0].message.content
    
    def _generate_groq(self, prompt: str) -> str:
        """Generate with Groq"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful GitLab onboarding assistant. Answer based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=500,
            top_p=1,
        )
        return response.choices[0].message.content
    
    def print_response(self, result: Dict, show_full_sources: bool = False):
        """Pretty print response"""
        print("\n" + "=" * 80)
        print(f"💬 {result['provider'].upper()} RAG RESPONSE")
        print("=" * 80)
        
        print(f"\n❓ QUERY:\n{result['query']}")
        print(f"\n💡 ANSWER:\n{result['answer']}")
        
        print(f"\n📚 SOURCES ({result['num_sources']}):")
        print("-" * 80)
        
        for doc in result['sources']:
            print(f"\n🔹 Source {doc['rank']}")
            print(f"   Title: {doc['metadata'].get('title', 'N/A')}")
            print(f"   Rerank Score: {doc.get('rerank_score', 0):.4f}")
            
            if show_full_sources:
                print(f"   Text: {doc['document']}")
            else:
                preview = doc['document'][:200] + "..." if len(doc['document']) > 200 else doc['document']
                print(f"   Preview: {preview}")
            
            print("-" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🧪 TESTING UNIVERSAL RAG PIPELINE")
    print("=" * 80)
    
    # Check for API keys (prioritize Groq)
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if groq_key:
        provider = "groq"
        print("\n✓ Using GROQ (Free & Fast!)")
    elif gemini_key:
        provider = "gemini"
        print("\n✓ Using GEMINI (Free!)")
    elif openai_key:
        provider = "openai"
        print("\n✓ Using OPENAI")
    else:
        provider = "groq"
        print("\n⚠️ No API keys found - retrieval-only mode")
        print("   Get free Groq key: https://console.groq.com/keys")
    
    print("=" * 80)
    
    # Get credentials path
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path is None:
        default_creds = Path(__file__).resolve().parents[3] / "mlops-476419-2c1937dab204.json"
        if default_creds.exists():
            creds_path = str(default_creds)
    
    # Initialize pipeline with GCS support
    rag = UniversalRAGPipeline(
        provider=provider,
        use_gcs=True,
        bucket_name="mlops-data-oa",
        project_id="mlops-476419",
        credentials_path=creds_path
    )
    
    test_queries = [
        "What is GitLab's approach to sustainability?",
        "How does risk management work at GitLab?",
        "Tell me about legal compliance"
    ]
    
    for query in test_queries:
        result = rag.generate_answer(query, k=3)
        rag.print_response(result)
        
        if rag.client:
            input("\n👉 Press Enter for next query...")
    
    print("\n" + "=" * 80)
    print("✅ UNIVERSAL RAG TEST COMPLETE!")
    print("=" * 80)
    print(f"\n✓ Provider: {provider.upper()}")
    print(f"✓ Model: {rag.model_name}")
    if provider in ["gemini", "groq"]:
        print("✓ Cost: $0.00 (FREE!)")
    print("=" * 80)