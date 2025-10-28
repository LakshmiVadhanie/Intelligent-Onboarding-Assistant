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

## Architecture

##  Prerequisites

- Python 3.10+
- Docker & Docker Compose
- PostgreSQL (for Airflow)
- OpenAI/Anthropic API key 
- Pinecone account 

## Installation

The steps for User installation are as follows:

1. Clone repository onto the local machine
```
git clone https://github.com/Thomas-George-T/Ecommerce-Data-MLOps.git
```
2. Check python version  >= 3.8
```python
python --version
```
3. Check if you have enough memory
```docker
docker run --rm "debian:bullseye-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'
```

<hr>

**FOR WINDOWS: Create a file called .env in the same folder as `docker-compose.yaml` and set the user as follows:**
```
AIRFLOW_UID=50000
```
**If you get the following error**
```
ValueError: Unable to configure handler 'processor'
```
**Setting the user manually like above fixes it**

<hr>

4. With Docker running, initialize the database. This step only has to be done once.
```docker
docker compose up airflow-init
```
5. Run airflow
```docker
docker-compose up
```
Wait until terminal outputs something similar to

`app-airflow-webserver-1  | 127.0.0.1 - - [17/Feb/2023:09:34:29 +0000] "GET /health HTTP/1.1" 200 141 "-" "curl/7.74.0"`

6. Visit localhost:8080 login with credentials

```
user:airflow2
password:airflow2
```
7. Run the DAG by clicking on the play button on the right side of the window

8. Stop docker containers
```docker
docker compose down
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
