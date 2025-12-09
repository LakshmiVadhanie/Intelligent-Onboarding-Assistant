# Quick Start Guide - Model Development Pipeline

## 🚀 Running the Complete Pipeline

### Prerequisites
- Python 3.8+
- 2GB RAM
- (Optional) OpenAI API key for LLM generation

---

## Option 1: Run Without OpenAI (Retrieval Only) ✅

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run End-to-End Test
```bash
python test_pipeline.py
```

**Expected Output:** `🎉 ALL TESTS PASSED! Pipeline is ready for submission!`

### Step 3: Test Retrieval
```bash
python -m src.retrieval.advanced_retriever
```

### Step 4: View Experiments
```bash
python -m mlflow ui --backend-store-uri file:./experiments/mlruns
```
Open: http://localhost:5000

---

## Option 2: Run With LLM Generation (Full RAG) 🤖

### Recommended: Google Gemini (FREE!) ⭐

**Step 1: Get Free API Key**
Visit: https://aistudio.google.com/app/apikey

**Step 2: Create .env file**
Create file `.env` in project root:
```
GOOGLE_API_KEY=your-key-here
```

**Step 3: Run RAG Pipeline**
```bash
python -m src.generation.rag_pipeline
```

**This will generate actual answers using Gemini 2.0 - completely FREE!**

---

### Alternative: OpenAI (Paid)

If you prefer OpenAI ($5 minimum):

**Step 1:** Get key from https://platform.openai.com/api-keys

**Step 2:** Add to `.env`:
```
OPENAI_API_KEY=your-key-here
```

**Step 3:** Modify provider in code to use OpenAI

**Cost:** ~$0.002 per query with GPT-3.5-turbo

### Step 3: Run RAG Pipeline
```bash
python -m src.generation.rag_pipeline
```

**This will generate actual answers using GPT-3.5-turbo!**

---

## 🐳 Running with Docker

### Build Image
```bash
docker build -t onboarding-assistant:latest .
```

### Run Tests in Container
```bash
docker run --rm onboarding-assistant:latest
```

### Run with API Key
```bash
docker run --rm -e OPENAI_API_KEY=your-key onboarding-assistant:latest
```

---

## 📊 Key Commands

| Task | Command |
|------|---------|
| Generate embeddings | `python -m src.embeddings.generate_embeddings` |
| Build vector store | `python -m src.retrieval.vector_store` |
| Test retrieval | `python -m src.retrieval.advanced_retriever` |
| Run evaluation | `python -m src.evaluation.metrics` |
| Check bias | `python -m src.evaluation.bias_detection` |
| RAGAS evaluation | `python -m src.evaluation.ragas_evaluator` |
| Track experiments | `python -m src.experiments.mlflow_tracking` |
| Full pipeline test | `python test_pipeline.py` |

---

## 🎯 Current Performance

Without OpenAI API (Retrieval-Only):
- **MRR:** 0.667
- **Precision@1:** 66.7%
- **Context Coverage:** 78.57%
- **No Bias Detected:** ✅
- **All Tests Passing:** 7/7

---

## ❓ Troubleshooting

**Q: Tests fail on first run?**  
A: Run embedding generation first: `python -m src.embeddings.generate_embeddings`

**Q: MLflow UI won't start?**  
A: Use: `python -m mlflow ui --backend-store-uri file:./experiments/mlruns`

**Q: Docker build fails?**  
A: Use `requirements-docker.txt` instead of full requirements

**Q: Want to enable LLM generation?**  
A: Set `OPENAI_API_KEY` environment variable (costs ~$0.50 for testing)

---

## 📧 Support

For issues, check:
1. README.md - Full documentation
2. DOCKER.md - Container deployment
3. GitHub Issues - Report problems
4. MLflow UI - View experiment history

---

**Last Updated:** November 12, 2025  
**Status:** Production Ready ✅