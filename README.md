# Intelligent Onboarding Assistant

<div align="center">

**An enterprise-grade RAG system that transforms company onboarding using GitLab's public knowledge base**

**[Live Demo](https://onboarding-ui-p5rleegxya-uc.a.run.app/)** • **[Setup Video](https://github.com/Mithun3110/Intelligent-Onboarding-Assistant/blob/main/MLOPS%20FINAL%20PRESENTATION.mp4)**

</div>

---

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [MLOps Pipeline](#mlops-pipeline)
- [Deployment](#deployment)
- [Testing](#testing)
- [License](#license)
- [Team](#team)

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Google Cloud account (for GCS storage)
- Groq API key (free tier: https://console.groq.com)
- Git
- Docker (optional, for containerized deployment)

### Complete Setup Guide

#### Step 1: Clone the Repository

```bash
git clone https://github.com/Mithun3110/Intelligent-Onboarding-Assistant.git
cd Intelligent-Onboarding-Assistant
```

#### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
# Navigate to Model_Pipeline directory
cd Model_Pipeline

# Install all required packages
pip install -r requirements.txt

# This will install:
# - sentence-transformers (for embeddings)
# - chromadb (vector database)
# - streamlit (web interface)
# - groq (LLM API)
# - google-cloud-storage (GCS integration)
# - mlflow (experiment tracking)
# - and other dependencies
```

#### Step 4: Set Up Google Cloud Storage (GCS)

**4.1: Create GCS Service Account**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to "IAM & Admin" → "Service Accounts"
3. Click "Create Service Account"
4. Name it (e.g., `mlops-storage-uploader`)
5. Grant roles: "Storage Admin" and "Storage Object Admin"
6. Click "Create Key" → Choose "JSON"
7. Save the JSON file in the project root directory

**4.2: Set Up GCS Bucket**

```bash
# Install Google Cloud SDK if not already installed
# Visit: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Create bucket (replace with your project ID)
gsutil mb -p your-project-id -l us-central1 gs://your-bucket-name

# Verify bucket
gsutil ls gs://your-bucket-name
```

#### Step 5: Set Up Groq API

**5.1: Get Groq API Key**

1. Visit [Groq Console](https://console.groq.com)
2. Sign up or log in
3. Navigate to "API Keys"
4. Click "Create API Key"
5. Copy the key (starts with `gsk_...`)

**5.2: Note about Groq Free Tier**

- 14,400 requests per day
- Rate limit: 30 requests per minute
- Models available: Mixtral, Llama 3, Gemma
- No credit card required

#### Step 6: Configure Environment Variables

Create a `.env` file in the `Model_Pipeline` directory:

```bash
# Navigate to Model_Pipeline if not already there
cd Model_Pipeline

# Create .env file
# On Windows:
copy .env.example .env
# On macOS/Linux:
cp .env.example .env
```

Edit the `.env` file and add the following:

```env
# ============================================
# API Keys
# ============================================

# Groq API (FREE - RECOMMENDED)
GROQ_API_KEY=your_groq_api_key_here

# Alternative: Google Gemini (FREE)
# GOOGLE_API_KEY=your_google_api_key_here
# GEMINI_API_KEY=your_google_api_key_here

# ============================================
# Google Cloud Storage Configuration
# ============================================

GCS_BUCKET_NAME=your_gcs_bucket_name
GCS_PROJECT_ID=your_gcp_project_id

# Google Cloud Credentials (use absolute paths)
# Windows example: C:\Users\YourName\Desktop\project\service-account.json
# macOS/Linux example: /home/username/project/service-account.json
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\project\your-service-account-key.json
GCS_KEY_PATH=C:\path\to\your\project\your-service-account-key.json

```

**Important Notes:**
- Replace `your_groq_api_key_here` with your actual Groq API key from Step 5
- Replace `your_gcs_bucket_name` with your GCS bucket name
- Replace `your_gcp_project_id` with your Google Cloud project ID
- For `GOOGLE_APPLICATION_CREDENTIALS` and `GCS_KEY_PATH`, use the **absolute path** to your service account JSON file
- **Groq is recommended** as the primary LLM provider (free tier, fast inference)
- Never commit the `.env` file or service account JSON to Git (they're in .gitignore)

#### Step 7: Verify Setup

```bash
# Test GCS connection
python -c "from google.cloud import storage; client = storage.Client(); print('GCS connection successful!')"

# Test Groq API
python -c "from groq import Groq; client = Groq(); print('Groq API connection successful!')"
```

#### Step 8: Initialize Data and Models

```bash
# Generate embeddings from GCS data
python src/embeddings/generate_embeddings.py

# Build vector store
python src/retrieval/vector_store.py

# This will take 5-10 minutes on first run
# Subsequent runs use cached embeddings
```

#### Step 9: Verify Everything Works

Run the comprehensive test suite:

```bash
# Run all tests
python test_pipeline.py

# Expected output:
# Data Loading: PASSED
# Embeddings Generation: PASSED
# Vector Store: PASSED
# Retrieval: PASSED
# RAG Pipeline: PASSED
# Evaluation Metrics: PASSED
```

### Troubleshooting Common Issues

**Issue 1: "ModuleNotFoundError: No module named 'google.cloud'"**
```bash
pip install google-cloud-storage
```

**Issue 2: "groq.AuthenticationError"**
- Check your `.env` file has the correct `GROQ_API_KEY`
- Verify the key starts with `gsk_`
- Make sure there are no extra spaces

**Issue 3: "GCS authentication failed"**
```bash
# Set environment variable manually
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-key.json"

# On Windows:
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your-service-account-key.json
```

**Issue 4: "ChromaDB collection not found"**
```bash
# Regenerate embeddings and vector store
python src/embeddings/generate_embeddings.py
python src/retrieval/vector_store.py
```

**Issue 5: Streamlit port already in use**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### Docker Setup (Alternative)

If you prefer containerized deployment:

```bash
# 1. Build the Docker image
docker build -t onboarding-assistant:latest -f Model_Pipeline/Dockerfile .

# 2. Run the container
docker run -p 8501:8501 \
  -e GROQ_API_KEY=your_groq_key \
  -v $(pwd)/your-service-account.json:/app/credentials.json \
  onboarding-assistant:latest

# On Windows PowerShell, use:
docker run -p 8501:8501 `
  -e GROQ_API_KEY=your_groq_key `
  -v ${PWD}/your-service-account.json:/app/credentials.json `
  onboarding-assistant:latest
```

### Expected Setup Time

- **First-time setup**: 15-20 minutes
  - Dependencies installation: 5 minutes
  - GCS setup: 3 minutes
  - Groq API setup: 2 minutes
  - Initial embeddings generation: 5-10 minutes

- **Subsequent runs**: Instant
  - Embeddings are cached
  - Vector store persists locally

---

## Overview

The **Intelligent Onboarding Assistant** is a production-ready Retrieval-Augmented Generation (RAG) system designed to accelerate employee onboarding by providing instant, accurate answers to company-related questions. Built using GitLab's comprehensive public documentation as a reference implementation, this system demonstrates enterprise-grade MLOps practices and can be adapted to any organization's knowledge base.

### Why This Project?

When new employees join a company, they're often overwhelmed by the sheer volume of documentation they need to absorb. At organizations like GitLab with 1000+ pages of handbook content, finding specific information can take 15-20 minutes per query. Traditional approaches like manually searching through documents, asking colleagues on Slack, or scheduling manager meetings are time-consuming and don't scale well.

Our solution addresses this by reducing query resolution time by 95%, from an average of 15-20 minutes down to just 1-2 seconds. Every response includes verifiable source citations, so users can trust the information they're getting and dive deeper if needed.

---

## Problem Statement

### Current Challenges

| Problem | Impact | Traditional Solution | Time Cost |
|---------|--------|---------------------|-----------|
| **Information Overload** | 1000+ handbook pages + scattered meeting recordings | Manual navigation & search | 15-20 min/query |
| **Inefficient Discovery** | Keyword search misses semantic queries | Read multiple documents | 10-30 min |
| **Meeting Content Hidden** | Video recordings not searchable | Watch entire meetings | Hours |
| **Senior Time Burden** | New hires interrupt experienced employees | Schedule manager meetings | 30-60 min |
| **Slow Ramp-Up** | Fragmented information sources | Trial and error | 4-6 weeks to productivity |

### Real-World Impact

- 80% of onboarding questions are repetitive and can be answered from existing documentation
- Senior employees spend over 10 hours per week answering routine questions from new hires
- New hires typically take 4-6 weeks to become fully productive due to information discovery bottlenecks

---

## Solution

### Key Innovation

1. **Hybrid Retrieval**: We combine semantic understanding through dense vector embeddings with keyword precision using BM25 sparse search. This dual approach handles both conceptual questions like "How does remote work function?" and exact queries like "What's the PTO policy?"

2. **Cross-Encoder Reranking**: After retrieving the top 20 candidate documents, we use a cross-encoder model to perform pairwise relevance scoring, refining the results down to the top 5 most relevant chunks. This additional step improves accuracy by about 30%.

3. **Multi-Modal Knowledge Integration**: Rather than just indexing static handbook pages, we also process meeting transcripts from YouTube videos and blog posts. This gives users access to discussions and decisions that might not be formally documented yet.

4. **Source Attribution**: Every generated answer includes direct citations with confidence scores and clickable links to the source documents. Users can verify information and explore context without taking our word for it.

5. **Continuous Learning**: When the system can't provide a satisfactory answer, we have a feedback loop that escalates to human experts and uses their responses to improve the knowledge base.

---

## Key Features

### Core Capabilities

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Hybrid Search** | Dense (MPNet) + Sparse (BM25) retrieval | Handles both conceptual and exact queries |
| **Cross-Encoder Reranking** | BGE-Reranker-Large for relevance refinement | 30% improvement in result quality |
| **Multi-Source Integration** | Handbook + Transcripts + Blogs | 3x broader knowledge coverage |
| **Sub-3s Response Time** | End-to-end pipeline optimization | 95% faster than manual search |
| **Perfect Retrieval** | MRR: 1.0, Precision@1: 100% | Always relevant top result |
| **Source Verification** | Direct links to source documents | Builds trust through transparency |
| **Free Deployment** | Groq API free tier | Zero inference costs |
| **GCS-Powered** | Google Cloud Storage for data | Scalable, versioned storage |

### MLOps Excellence

- **Automated Pipelines**: Apache Airflow DAGs for weekly data updates
- **Experiment Tracking**: MLflow for model comparison and versioning
- **Bias Detection**: Fairlearn-based analysis across demographic slices
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Monitoring**: Comprehensive logging and performance metrics
- **Data Versioning**: DVC for reproducible experiments

---

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Data Storage** | Google Cloud Storage, ChromaDB, DVC |
| **Orchestration** | Apache Airflow, Docker Compose |
| **Embeddings** | Sentence-Transformers (MPNet, BGE-Large) |
| **Retrieval** | ChromaDB (Dense), Elasticsearch (Sparse), BGE-Reranker |
| **Generation** | Groq (Mixtral-8x7B, Llama 3) |
| **Monitoring** | MLflow, Prometheus, Grafana, RAGAS |
| **CI/CD** | GitHub Actions, Docker|
| **Frontend** | Streamlit, FastAPI |

---

## Project Structure

```
Intelligent-Onboarding-Assistant/
├── Data-Ingestion-GCS/          # Data collection and GCS upload
│   ├── scraper.py               # GitLab handbook scraper
│   ├── transcription.py         # YouTube video transcription
│   ├── preprocess.py            # Text cleaning and normalization
│   ├── validate_data.py         # Great Expectations validation
│   ├── bias_detection.py        # Fairlearn bias analysis
│   └── upload_to_gcs.py         # GCS bucket operations
│
├── Data_Pipeline/               # Airflow orchestration
│   ├── dags/
│   │   └── data_pipeline_dag.py # Automated weekly updates
│   ├── docker-compose.yml       # Airflow services
│   └── requirements.txt         # Pipeline dependencies
│
├── Model_Pipeline/              # Core RAG implementation
│   ├── src/
│   │   ├── embeddings/
│   │   │   └── generate_embeddings.py  # MPNet embeddings
│   │   ├── retrieval/
│   │   │   ├── vector_store.py         # ChromaDB operations
│   │   │   ├── retriever.py            # Baseline retrieval
│   │   │   └── advanced_retriever.py   # Hybrid + reranking
│   │   ├── generation/
│   │   │   └── rag_pipeline.py         # End-to-end RAG
│   │   ├── evaluation/
│   │   │   ├── metrics.py              # P@K, R@K, MRR, NDCG
│   │   │   ├── ragas_evaluator.py      # Faithfulness, relevancy
│   │   │   ├── bias_detection.py       # Fairness analysis
│   │   │   └── sensitivity_analysis.py # Hyperparameter tuning
│   │   └── experiments/
│   │       ├── mlflow_tracking.py      # Experiment logging
│   │       └── model_registry.py       # Version management
│   │
│   ├── app.py                   # Streamlit web interface
│   ├── test_pipeline.py         # Unit tests
│   ├── test_gcs_pipeline.py     # GCS integration tests
│   ├── Dockerfile               # Container definition
│   └── requirements.txt         # Python dependencies
│
├── experiments/                 # MLflow tracking
│   ├── mlruns/                  # Experiment logs
│   ├── retrieval_evaluation.json
│   ├── ragas_evaluation.json
│   ├── bias_report.json
│   └── sensitivity_analysis.json
│
├── models/                      # Trained models
│   ├── embeddings/              # Generated embeddings
│   ├── vector_store/            # ChromaDB database
│   └── registry/                # Model versions
│
├── .github/workflows/           # CI/CD automation
│   ├── ci-pipeline.yml          # Testing and validation
│   └── tests.yml                # Automated test suite
│
├── .env.example                 # Environment template
└── README.md                    # This file
```

---

## MLOps Pipeline

### Data Pipeline

```bash
# Automated weekly via Airflow DAG
airflow dags trigger data_pipeline_dag

# Manual execution
cd Data-Ingestion-GCS
python run_bias_pipeline.py
```

**Pipeline Stages**:
1. **Scraping**: GitLab handbook + YouTube transcripts
2. **Preprocessing**: Cleaning, normalization, deduplication
3. **Validation**: Great Expectations schema checks
4. **Bias Detection**: Fairlearn analysis across demographics
5. **Upload**: GCS bucket storage with versioning
6. **Monitoring**: Airflow alerts for anomalies

### Model Development

```bash
# 1. Generate embeddings
cd Model_Pipeline
python src/embeddings/generate_embeddings.py

# 2. Build vector store
python src/retrieval/vector_store.py

# 3. Run evaluation
python src/evaluation/metrics.py

# 4. RAGAS assessment
python src/evaluation/ragas_evaluator.py

# 5. Bias analysis
python src/evaluation/bias_detection.py

# 6. Sensitivity analysis
python src/evaluation/sensitivity_analysis.py
```

### Experiment Tracking

```bash
# Start MLflow UI
cd Model_Pipeline
mlflow ui --host 0.0.0.0 --port 5000
```

Open http://localhost:5000 to view:
- Model comparisons
- Hyperparameter tuning
- Metric evolution
- Artifact storage

---

## Deployment

### Local Development

```bash
cd Model_Pipeline
streamlit run app.py
```

### Docker Deployment

```bash
# Build image
docker build -t onboarding-assistant:latest -f Model_Pipeline/Dockerfile .

# Run container
docker run -d -p 8501:8501 \
  -e GROQ_API_KEY=${GROQ_API_KEY} \
  -v $(pwd)/service-account-key.json:/app/credentials.json \
  --name onboarding-app \
  onboarding-assistant:latest

# View logs
docker logs -f onboarding-app
```

### Cloud Deployment (GCP)

Deploy to Google Cloud Run:

```bash
# Authenticate
gcloud auth login

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/your-project-id/onboarding-assistant

# Deploy to Cloud Run
gcloud run deploy onboarding-assistant \
  --image gcr.io/your-project-id/onboarding-assistant \
  --platform managed \
  --region us-central1 \
  --set-env-vars GROQ_API_KEY=${GROQ_API_KEY} \
  --allow-unauthenticated
```

**Live Demo:** https://onboarding-ui-p5rleegxya-uc.a.run.app/

---

## Testing


### Integration Tests

```bash
# Navigate to Model_Pipeline directory
cd Model_Pipeline

# Test full pipeline
python test_pipeline.py

# Test GCS integration
python test_gcs_pipeline.py

```

# MLOps Monitoring Pipeline

Production models require continuous monitoring to detect performance degradation and data drift. We implement a 5-step automated monitoring system:

## Step 1: Metrics Collection
Collects **11 key metrics per query**: Precision@5, Recall@5, F1@5, MRR, NDCG@5, Response Time, Semantic Relevance, Query Count, Embedding Dimensions.

```python
from src.monitoring.metrics_collector import MetricsCollector

collector = MetricsCollector()
metrics = collector.collect_metrics(
    query="What is the onboarding process?",
    retrieved_docs=[doc1, doc2, doc3, doc4, doc5],
    generated_response="...",
    response_time=0.45,
    relevant_doc_indices=[0, 1, 3]
)
daily_metrics = collector.get_aggregated_metrics(window='1d')
```

## Step 2: Data Shift Detection
Uses **Evidently AI** (distribution comparison) and **TFDV** (schema validation) to detect drift.

```python
from src.monitoring.data_shift_detector import DataShiftDetector

detector = DataShiftDetector()
drift_report = detector.detect_drift(
    new_data=new_data,
    reference_data=original_training_data,
    methods=['evidently', 'tfdv']
)
print(f"Data Drift: {drift_report['drift_percentage']}%")
```

## Step 3: Threshold Management
**18 configurable thresholds** across performance (Precision>0.5, F1>0.35, Response<5s) and drift metrics (Evidently>30%, TFDV>20%).

```python
from src.monitoring.threshold_manager import ThresholdManager

manager = ThresholdManager()
violations = manager.check_thresholds(metrics)
if violations:
    print(f"CRITICAL: {violations} → TRIGGER RETRAINING")
```

## Step 4: Automated Retraining Pipeline
**5-step pipeline**: DataPuller → ModelRetrainer → ModelValidator → ModelComparator → ModelDeployer. Deploys if improvement ≥2%.

```python
from src.monitoring.retraining_pipeline import RetrainingPipeline

pipeline = RetrainingPipeline(data_source='gcs')
result = pipeline.run_full_pipeline(
    trigger_reason="Precision dropped to 0.42",
    data_source_path="gs://bucket/data.json"
)
# Deploys v2 if improvement_percentage >= 2%
```

## Step 5: Stakeholder Notifications
**Email + Slack** notifications for: Retraining Triggered, Model Deployed, Deployment Held, Failures.

```python
from src.monitoring.notifications import NotificationManager

manager = NotificationManager(
    email_recipients=['stakeholder@company.com'],
    sender_email='alerts@company.com',
    sender_password='app_password'
)
manager.notify_retraining_triggered(
    trigger_reason="Precision@5 dropped from 0.55 to 0.42",
    drift_percentage=45.0
)
```


## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


## Team

**Team 13 - Northeastern University MLOps Course**

- Saran Jagadeesan Uma
- Lakshmi Vandhanie Ganesh
- Mithun Dineshkumar
- Zankhana Pratik Mehta
- Akshaj Nevgi

---

## Acknowledgments

- **GitLab** for providing comprehensive public documentation
- **Groq** for fast, free LLM inference
- **Google Cloud** for GCS storage and deployment
- **Northeastern University** for MLOps course structure
- **Open-source community** for amazing tools

---

## Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Mithun3110/Intelligent-Onboarding-Assistant/issues)

<div align="center">

**Made by Team 13 | Northeastern University**

[Back to Top](#intelligent-onboarding-assistant)

</div>
