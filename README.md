# Intelligent Onboarding Assistant

[Lakshmi Vandhanie Ganesh](https://github.com/LakshmiVadhanie),
[Zankhana Pratik Mehta](https://github.com/zankhana46),
[Mithun Dineshkumar](https://github.com/Mithun3110),
[Saran Jagadeesan Uma](https://github.com/Saran-Jagadeesan-Uma),
[Akshaj Nevgi](https://github.com/Akshaj-N)

# Introduction 
The Intelligent Onboarding Assistant is a RAG (Retrieval-Augmented Generation) system designed to transform the employee onboarding experience by consolidating fragmented organizational knowledge into a conversational AI assistant. New employees typically face information overload, struggling to navigate thousands of pages of documentation and scattered meeting recordings to understand company culture, processes and policies. The project addresses these challenges by implementing a hybrid retrieval system that combines semantic search with keyword matching across multiple data sources and reducing the average query resolution time and cutting onboarding duration.

This project demonstrates a complete MLOps pipeline encompassing automated data ingestion from multiple sources including company handbooks and meeting transcripts, intelligent chunking strategies, multi-modal knowledge integration and production-ready deployment infrastructure with continuous monitoring. The objectives of this project include preparing the data pipeline, implementing hybrid retrieval with cross-encoder reranking, establishing comprehensive evaluation frameworks using RAGAS metrics, deploying a containerized FastAPI service with CI/CD automation and demonstrating real-world production viability. Thus, this project serves as a practical demonstration of how modern MLOps practices combined with RAG can dramatically reduce onboarding time, decrease senior employee mentorship burden and ultimately transform the employee experience while delivering measurable business value in the competitive talent landscape.


# Dataset Information 

The dataset contains GitLab's public documentation ecosystem which serves as a real-world proxy for enterprise onboarding materials. GitLab was selected as the project's data source due to its comprehensive handbook content (1000+ pages), publicly accessible meeting recordings, complete operational transparency and zero data access barriers. This combination provides the multi-modal and real-world complexity necessary to build and validate a robust onboarding assistant while avoiding legal and privacy constraints.

## Data Card
  
| **Attribute**       | **Details**                           |
|---------------------|---------------------------------------|
| **Dataset Name**     | GitLab-Onboarding-Knowledge-v1        |
| **Total Size**       | ~800MB - 1.5GB                        |
| **Source**           | [GitLab Handbook](https://handbook.gitlab.com/) |
| **Document Count**   | 1000+ handbook pages, 50+ videos      |
| **Format**           | Markdown (handbook), MP4/transcripts (meetings) |
| **Access**           | Public (CC BY-SA 4.0 license)         |


## Data Sources

1. **GitLab Handbook**: [GitLab Handbook](https://handbook.gitlab.com/)
   - Web scraping via BeautifulSoup/Scrapy
   - Markdown files from GitLab's public repository

2. **Meeting Recordings**: [YouTube GitLab Channel](https://www.youtube.com/@Gitlab)
   - Video URLs: [GitLab YouTube](https://www.youtube.com/@Gitlab)
   - Transcription: YouTube API + Whisper (for non-transcribed videos)

3. **GitLab Blog**: [GitLab Blog](https://about.gitlab.com/blog/)
   - RSS feed parsing
   - Filtered for onboarding-relevant topics

##  Prerequisites

- Python 3.9+

## Installation

The steps for User installation are as follows:

#### Step 1: Clone repository
```
git clone https://github.com/LakshmiVadhanie/Intelligent-Onboarding-Assistant.git
cd Intelligent-Onboarding-Assistant
```
Check python version  >= 3.9
```python
python --version
```

#### Step 2: Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 3: Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
#### Step 4: Environment Setup
```bash
# Initialize Airflow database
airflow db init

# Start Airflow webserver (in one terminal)
airflow webserver --port 8080

# Start Airflow scheduler (in another terminal)
airflow scheduler
```

### Step 5: Initialize DVC

```bash
# Initialize DVC
dvc init

# Configure remote storage
dvc remote add -d myremote /path/to/dvc/storage

# Pull existing data (if available)
dvc pull
```

### Step 6: Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check Airflow DAGs
airflow dags list
```
---

## Project Structure
```
Intelligent-Onboarding-Assistant/
├── README.md                        # This file
├── fix_airflow_setup.sh             # Helper script for local Airflow fixes
├── test_dag_local.py                # Small harness to test DAG logic locally
├── test_scraper.py                  # Tests for scraping/ingestion
├── test_setup.py                    # Setup/unit test helpers
├── data/                            # Data artifacts and outputs
│   ├── anomaly_report.json
│   └── pipeline_statistics.json
├── data-pipeline/                   # DVC metadata and pipeline requirements
│   ├── data.dvc
│   └── requirements.txt
├── airflow/                         # Airflow configuration + dags + logs
│   ├── airflow.cfg
│   ├── webserver_config.py
│   ├── dags/
│   │   └── main_pipeline_dag.py     # Primary pipeline DAG
│   └── logs/
├── configs/
│   └── pipeline_config.yaml
├── logs/                            # Pipeline/runtime logs
├── monitoring/
│   └── dashboards/
├── scripts/                         # Modular pipeline code
│   ├── ingestion/
│   │   ├── blog_fetcher.py
│   │   ├── gitlab_scraper.py
│   │   ├── v1.py
│   │   └── video_extractor.py
│   ├── preprocessing/
│   │   ├── chunking_strategy.py
│   │   ├── meeting_transcript_cleaner.py
│   │   └── transcript_cleaner.py
│   ├── monitoring/
│   │   └── alert_manager.py
│   ├── utils/
│   │   ├── config_loader.py
│   │   ├── gcs_uploader.py
│   │   ├── logging_config.py
│   │   └── logging_setup.py
│   └── validation/
│       ├── bias_detector.py
│       ├── data_validator.py
│       └── fairness_analysis.py
└── tests/                           # pytest unit tests
  ├── conftest.py
  ├── test_preprocessing/
  │   └── test_transcript_cleaner.py
  └── test_validation/
    ├── test_bias_detector.py
    ├── test_data_validator.py
    └── test_fairness_analysis.py
```

## Data Pipeline

Our data pipeline is modularized right from data ingestion to preprocessing to make our data ready for modeling. It is made sure that every module functions as expected by following Test Driven Development (TDD). This is achieved through enforcing tests for every module. 

We utilize Apache Airflow for our pipeline. We create a DAG with our modules.

![DAG Image](assets/dag.jpg "Airflow DAG")
Pictured: Our Airflow DAG

The following is the explanation of our Data pipeline DAG

## Data Pipeline Components

The data pipeline in this repository is modular and orchestrated by Airflow (see `airflow/dags/main_pipeline_dag.py`). Each module is implemented as a small, testable script under `scripts/` and is executed as a task in the DAG. The stages below map the common pipeline components to the actual scripts and files in this repo.

### 1. Data acquisition / downloading
The first stage involves fetching raw content from sources and persist to the local `data/` folder (or a DVC-tracked store).
- Relevant scripts:
  - `scripts/ingestion/gitlab_scraper.py`: scraper for handbook/documentation pages.
  - `scripts/ingestion/blog_fetcher.py`: fetches and parses blog posts or RSS feeds.
  - `scripts/ingestion/video_extractor.py`: collects video URLs and metadata (used with transcription tools).
  - `scripts/ingestion/v1.py`: ingestion entrypoints / orchestration helper for ingestion flows.

### 2. Data cleaning & preprocessing
The next phase involves cleaning raw text and transcript artifacts, normalizing formats and preparing downstream inputs.
- Relevant scripts:
  - `scripts/preprocessing/transcript_cleaner.py`: main transcript/text cleaning utilities.
  - `scripts/preprocessing/meeting_transcript_cleaner.py`: meeting-specific transcript cleaning and normalization.
  - `scripts/preprocessing/chunking_strategy.py`: document chunking logic used to split large documents into retrieval-friendly pieces.

Notes: by convention ingestion scripts write raw artifacts to `data/raw/` (or `data/`) and preprocessing writes cleaned outputs to `data/processed/` (or a repo `data/` path). When running via Airflow the DAG tasks pass file locations or use a shared configuration in `configs/pipeline_config.yaml` and `scripts/utils/config_loader.py`.

### 3. Validation, anomaly detection & statistics
This phase validates schema, detects anomalies and computes dataset statistics or reports.
- Relevant scripts:
  - `scripts/validation/data_validator.py` — validators and schema checks.
  - `scripts/validation/bias_detector.py` — performs data slicing and bias checks.
  - `scripts/validation/fairness_analysis.py` — fairness reporting and slice-level metrics.

Outputs: anomaly reports and pipeline statistics are written to `data/anomaly_report.json` and `data/pipeline_statistics.json` which can be monitored and used to trigger alerts via `scripts/monitoring/alert_manager.py` or Airflow alert hooks.

### 4. Feature engineering & downstream artifacts
Thus step derives features, perform chunking/embedding-creation and produce artifacts used by retrieval or modeling.
- Relevant script:
  - `scripts/preprocessing/chunking_strategy.py` is used to create chunks for the retrieval layer.
  - Embedding generation and storage may be implemented in `scripts/utils/` or in downstream model code that consumes cleaned/processed data.

### 5. Orchestration (Airflow DAG)
- Purpose: wire tasks together, handle retries, scheduling and logging.
- The primary DAG is `airflow/dags/main_pipeline_dag.py`. Typical DAG task mapping:
  - Ingestion task(s) -> call to `scripts/ingestion/*`
  - Preprocessing task(s) -> call to `scripts/preprocessing/*`
  - Validation & statistics task(s) -> call to `scripts/validation/*`
  - Monitoring/alerting -> call to `scripts/monitoring/alert_manager.py`

Each DAG task should be idempotent and write outputs to deterministic paths (so DVC can track them and CI can assert reproducibility).

### 6. Data versioning & model tracking
- Data produced by the pipeline should be recorded with DVC (`data-pipeline/data.dvc`) and models/experiments tracked with MLflow. Use `dvc add` for large artifacts and `mlflow` APIs to log experiments.

Example: run an ingestion + preprocess pair locally (manual/debug run):

```bash
source .venv/bin/activate
python scripts/ingestion/v1.py         # run ingestion step (entrypoint)
python scripts/preprocessing/transcript_cleaner.py   # run local preprocessing
python scripts/validation/data_validator.py         # optional validation run
```

When running via Airflow, enable and trigger the DAG in the web UI or run `airflow dags trigger <dag_id>` to launch the full pipeline.

---

# Contributing / Development Guide

**This is the user guide for developers**

Before developing our code, we should install the required dependencies
```python
pip install -r requirements.txt
```

## Testing
Before pushing code to GitHub, Run the following commands locally to ensure build success. Working on the suggestions given by `Pylint` improves code quality. Ensuring that the test cases are passed by `Pytest` are essential for code reviews and maintaining code quality.

To test for formatting and code leaks, run the following:
```python
pytest --pylint
```

To running the test suites for the modules, run the following:
```python
pytest 
```
## DVC

Steps to initialize and track files using DVC

1. Initialize dvc in the parent directory of your local repository.
    ```python
    dvc remote add -d temp /tmp/dvcstore
    ```
2. Set up remote bucket.
    ```python
    dvc remote add -d temp /tmp/dvcstore
    ```
3. Add the location as default to your remote bucket.
    ```python
    dvc remote add -d myremote gs://<mybucket>/<path>
    ```
4. Don't forget to modify your credentials.
    ```python
    dvc remote modify --lab2 credentialpath <YOUR JSON TOKEN>```

### License
This project is licensed under the MIT License. See LICENSE file for details.
