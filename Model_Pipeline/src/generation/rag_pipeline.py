import google.generativeai as genai
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
                 provider: str = "gemini", 
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 0.3):
        logger.info(f"Initializing Universal RAG Pipeline with {provider.upper()}...")
        
        logger.info("Setting up advanced retriever with reranking...")
        self.retriever = AdvancedRetriever()
        
        self.provider = provider.lower()
        self.temperature = temperature
        self.client = None
        self.model_name = model
        
        if self.provider == "gemini":
            self._init_gemini(api_key, model)
        elif self.provider == "openai":
            self._init_openai(api_key, model)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'openai'")
        
        logger.info(" Universal RAG Pipeline initialized!")
    
    def _init_gemini(self, api_key: Optional[str], model: Optional[str]):
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if api_key is None:
            logger.warning("  No Google API key found!")
            logger.warning("Get free key: https://aistudio.google.com/app/apikey")
            logger.warning("Set: $env:GOOGLE_API_KEY='your-key'")
            logger.warning("Pipeline will only retrieve, not generate")
            return
        
        genai.configure(api_key=api_key)
        self.model_name = model or "gemini-2.0-flash"
        self.client = genai.GenerativeModel(self.model_name)
        logger.info(f" Gemini initialized: {self.model_name} (FREE!)")
    
    def _init_openai(self, api_key: Optional[str], model: Optional[str]):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key is None:
            logger.warning("  No OpenAI API key found!")
            logger.warning("Set: $env:OPENAI_API_KEY='your-key'")
            logger.warning("Pipeline will only retrieve, not generate")
            return
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model or "gpt-3.5-turbo"
        logger.info(f" OpenAI initialized: {self.model_name}")
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict]:
        return self.retriever.retrieve(query, k=k)
    
    def build_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            title = doc['metadata'].get('title', 'Untitled')
            text = doc['document']
            context_parts.append(f"[Source {i}: {title}]\n{text}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are an intelligent onboarding assistant for GitLab. Answer the question based ONLY on the provided context from GitLab's handbook and meeting transcripts.

If the answer is not in the context, say "I don't have enough information to answer this based on the available documents."

Be concise, accurate, and cite which source(s) you're using (e.g., "According to Source 1...").

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_answer(self, query: str, k: int = 5) -> Dict:

        logger.info("=" * 80)
        logger.info(f"Processing query: {query}")
        logger.info("=" * 80)
        
        retrieved_docs = self.retrieve_context(query, k=k)
        logger.info(f" Retrieved {len(retrieved_docs)} documents with reranking")
        
        if self.client is None:
            logger.warning("  No LLM client - returning retrieval only")
            return {
                'query': query,
                'answer': "[Generation not available - no API key set]",
                'sources': retrieved_docs,
                'num_sources': len(retrieved_docs),
                'provider': self.provider,
                'model': None
            }
        
        prompt = self.build_prompt(query, retrieved_docs)
        logger.info(" Built prompt with context")
        
        logger.info(f"Generating answer using {self.provider.upper()}...")
        
        try:
            if self.provider == "gemini":
                answer = self._generate_gemini(prompt)
            else: 
                answer = self._generate_openai(prompt)
            
            logger.info(" Answer generated successfully")
            
        except Exception as e:
            logger.error(f" Error generating answer: {e}")
            answer = f"[Error: {str(e)}]"
        
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
        response = self.client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=500,
            )
        )
        return response.text
    
    def _generate_openai(self, prompt: str) -> str:
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
    
    def print_response(self, result: Dict, show_full_sources: bool = False):
        print("\n" + "=" * 80)
        print(f"💬 {result['provider'].upper()} RAG RESPONSE")
        print("=" * 80)
        
        print(f"\n QUERY:\n{result['query']}")
        print(f"\n ANSWER:\n{result['answer']}")
        
        print(f"\n SOURCES ({result['num_sources']}):")
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
    print(" TESTING UNIVERSAL RAG PIPELINE")
    print("=" * 80)
    
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if gemini_key:
        provider = "gemini"
        print("\n Using GEMINI (Free!)")
    elif openai_key:
        provider = "openai"
        print("\n Using OPENAI")
    else:
        provider = "gemini"
        print("\n  No API keys found - retrieval-only mode")
    
    print("=" * 80)
    
    rag = UniversalRAGPipeline(provider=provider)
    
    test_queries = [
        "What is GitLab's approach to sustainability?",
        "How does risk management work at GitLab?",
        "Tell me about legal compliance"
    ]
    
    for query in test_queries:
        result = rag.generate_answer(query, k=3)
        rag.print_response(result)
        
        if rag.client:
            input("\n  Press Enter for next query...")
    
    print("\n" + "=" * 80)
    print(" UNIVERSAL RAG TEST COMPLETE!")
    print("=" * 80)
    print(f"\n Provider: {provider.upper()}")
    print(f" Model: {rag.model_name}")
    if provider == "gemini":
        print(" Cost: $0.00 (FREE!)")
    print("=" * 80)