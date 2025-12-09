# 🧠 Intelligent Onboarding Assistant — MLOps Data Pipeline

## 📘 Overview
This repository contains the **Intelligent Onboarding Assistant Data Pipeline**, developed as part of the MLOps course project.

The goal of this project is to automate the **ingestion, validation, and fairness analysis** of onboarding materials from GitLab — combining both **web documentation** and **meeting transcripts**.  
The pipeline detects potential **bias** in text content and mitigates it by generating debiased versions of all processed data.

By the end of the pipeline:
- All text data (from handbook + YouTube meetings) is **scraped, validated, analyzed and debiased**
- Bias metrics are stored in structured JSON files
- Results are orchestrated, logged and monitored via **Apache Airflow**

---

## 🧩 Architecture

All components are orchestrated via Airflow’s DAG:

[scrape_handbook] → [transcribe_youtube] → [validate_data] → [bias_detection] → [bias_mitigation] → [notify_success]


---

## ⚙️ Pipeline Components

### 1️⃣ **Data Acquisition**
**Scripts:**  
- `scraper.py` — Scrapes structured text content from [GitLab’s Handbook](https://handbook.gitlab.com/).  
- `transcription.py` — Downloads and transcribes meeting videos from a YouTube playlist using **OpenAI Whisper**.

**Outputs:**  
- `data/handbook_paragraphs.json`  
- `data/meeting_transcripts/all_transcripts.json`

Both scripts integrate with the shared `ParagraphPreprocessor` for text normalization.

---

### 2️⃣ **Data Validation**
**Script:** `validate_data.py`  
- Validates that key fields exist (`title`, `paragraph`, `transcript`, etc.)  
- Checks JSON structure and non-empty content  
- Exits with `code=1` if validation fails, so Airflow marks the DAG task as failed

**Output:**  
Validation summary logged to Airflow and console.

---

### 3️⃣ **Bias Detection**
**Script:** `bias_detection.py`  
- Scans both datasets for sensitive or biased words using category-based lexicons  
- Categories: gender, ethnicity, age, ability, religion  
- Generates structured `bias_report.json` with per-record counts

**Output:**  
`data/bias_report.json`

---

### 4️⃣ **Bias Mitigation**
**Script:** `bias_mitigation.py`  
- Reads the bias report and applies neutral replacements for biased terms  
- Creates clean, debiased versions of all datasets

**Output:**  
- `data/debiased_data/handbook_paragraphs_debiased.json`  
- `data/debiased_data/all_transcripts_debiased.json`

---

### 5️⃣ **Pipeline Orchestration (Airflow)**
**File:** `data_pipeline_dag.py`  
- Defines the full DAG using BashOperators and EmailOperator  
- Automates end-to-end pipeline execution  
- Sends email notifications upon success/failure  
- Logs all stages centrally in Airflow UI  

**Task Order:**

scrape_handbook → transcribe_youtube → validate_data → bias_detection → bias_mitigation → notify_success

## 🧱 Folder Structure

📂 Intelligent-Onboarding-Assistant
```
│
├── dags/
│ ├── scripts/
│ │ ├── scraper.py
│ │ ├── transcription.py
│ │ ├── preprocess.py
│ │ ├── validate_data.py
│ │ ├── bias_detection.py
│ │ ├── bias_mitigation.py
│ │ ├── logging_utils.py
│ │ └── tests/
│ └── data_pipeline_dag.py
│
├── data/
│ ├── handbook_paragraphs.json
│ ├── meeting_transcripts/all_transcripts.json
│ ├── bias_report.json
│ └── debiased_data/
│
├── logs/
│
├── dvc.yaml
├── requirements.txt
├── Dockerfile
└── README.md

```
---

## 🧠 Key Features
- ✅ **Automated ingestion** from multiple data sources (web + video)
- ✅ **Schema validation & anomaly detection**
- ✅ **Lexicon-based bias detection**
- ✅ **Bias mitigation via neutral replacements**
- ✅ **Version control with DVC**
- ✅ **Centralized logging**
- ✅ **Email alerting and failure handling**
- ✅ **Fully orchestrated Airflow DAG**

---

## 🧩 Technology Stack
| Layer | Tools Used |
|-------|-------------|
| **Orchestration** | Apache Airflow |
| **Data Versioning** | DVC |
| **Data Processing** | Python (BeautifulSoup, Whisper) |
| **Bias Analysis** | Custom lexicon-based detector |
| **Monitoring** | Airflow Logs, Email Alerts |
| **Deployment** | Dockerized environment |
| **Testing** | `unittest`, custom test scripts |

---

## 🧾 Evaluation Criteria Mapping

| **Criterion** | **How It’s Addressed** |
|----------------|------------------------|
| **Documentation** | Well-commented scripts, README, and logs |
| **Modularity** | Each step is an independent Python module |
| **Airflow DAG** | `data_pipeline_dag.py` with sequential dependencies |
| **Logging & Tracking** | Shared `logging_utils.py` across all scripts |
| **Data Version Control** | All datasets tracked via `.dvc` files |
| **Pipeline Optimization** | Lightweight, modular scripts with fail-fast validation |
| **Schema Validation** | Implemented in `validate_data.py` |
| **Anomaly & Alerting** | Airflow email alerts + validation exits |
| **Bias Detection & Mitigation** | Lexical scanning + replacements; report and debiased data generated |
| **Test Modules** | Unit tests under `/tests` |
| **Reproducibility** | Dockerized setup + relative paths |
| **Error Handling** | Try/except in all scripts, Airflow retries disabled for deterministic runs |

---

## 📦 Setup & Execution

1. Clone Repository
```bash
git clone https://github.com/<your-username>/Intelligent-Onboarding-Assistant.git
cd Intelligent-Onboarding-Assistant
```
2. Install Dependencies
```
pip install -r requirements.txt
```
3. Initialize DVC
```
dvc init
dvc pull   # if remote data is configured
```
4. Start Airflow
```
docker-compose up
```
5. Trigger Pipeline

In the Airflow UI:

Navigate to DAGs → data_pipeline_dag

Click Trigger DAG

Monitor progress in the Gantt view or logs
