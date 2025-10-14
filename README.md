# Intelligent-Onboarding-Assistant

An intelligent RAG-based onboarding assistant that processes GitLab's public handbook and meeting recordings to provide instant answers to new employee questions. This project demonstrates end-to-end MLOps practices including automated data pipelines, hybrid retrieval, evaluation metrics, and production deployment.

## Project Overview

New employees face information overload when navigating 1000+ pages of company documentation and meeting recordings. This system consolidates GitLab's publicly available handbook and YouTube meeting transcripts into a conversational assistant that reduces query resolution time from 15-20 minutes to under 2 minutes.

Project Status
In Development - This project is currently being built as part of an MLOps course.

**Dataset**: GitLab Public Handbook + YouTube Meeting Transcriptions

## Architecture



## Technology Stack

- Orchestration: Apache Airflow
- Embeddings: OpenAI text-embedding-3-large / Sentence-BERT
- Vector Database: Pinecone
- LLM: Claude 3.5 / GPT-4
- API: FastAPI
- Evaluation: RAGAS framework
- Monitoring: MLflow, Prometheus, Grafana
- Deployment: Docker

## Key Features
1. **Hybrid Retrieval System**
- Combines semantic search (dense vectors) with keyword search (BM25)
- Cross-encoder reranking for improved relevance
- Handles both exact queries and conceptual questions

2. **Multi-Modal Knowledge Base**
- GitLab handbook (1000+ pages of markdown)
- Meeting transcripts from YouTube (50+ videos via Whisper API)
- Unified search across documents and discussions

3. **Source Attribution**
- Every answer includes clickable citations
- Direct links to handbook sections or meeting timestamps
- Confidence scores for transparency

4. **MLOps Pipeline**
- Automated data ingestion via Airflow DAGs
- RAGAS evaluation (faithfulness, answer relevancy, context precision)
- Experiment tracking with MLflow
- Real-time monitoring with Prometheus/Grafana


##  Prerequisites

- Python 3.10+
- Docker & Docker Compose
- PostgreSQL (for Airflow)
- OpenAI/Anthropic API key 
- Pinecone account 

## Installation

### 1. Clone Repository
```bash
  # Clone repository
  git clone https://github.com/your-username/Intelligent-Onboarding-Assistant.git
  cd Intelligent-Onboarding-Assistant
  
  # Create virtual environment
  python -m venv venv
  source venv/bin/activate  # Windows: venv\Scripts\activate
  
  # Install dependencies
  pip install -r requirements.txt
  
  # Set up environment variables
  cp .env.example .env
  # Edit .env with your API keys

```

## Project Structure
```
Intelligent-Onboarding-Assistant/
├── data/                   # Data storage (gitignored)
│   ├── raw/                # Scraped handbook & transcripts
│   ├── processed/          # Chunked documents
│   └── sample/             # Sample test data
├── src/                    # Source code
│   ├── data/               # Data ingestion & preprocessing
│   ├── embeddings/         # Embedding generation
│   ├── retrieval/          # Hybrid search & reranking
│   ├── generation/         # LLM generation
│   ├── evaluation/         # Metrics & RAGAS
│   └── api/                # FastAPI application
├── airflow/                # Airflow DAGs
├── notebooks/              # Exploratory analysis
├── tests/                  # Unit & integration tests
├── configs/                # Configuration files
├── scripts/                # Automation scripts
└── docs/                   # Documentation
```
### Team

Lakshmi Vandhanie Ganesh, 
Zankhana Pratik Mehta, 
Mithun Dineshkumar,
Saran Jagadeesan Uma, 
Akshaj Nevgi

### License
This project is licensed under the MIT License. See LICENSE file for details.