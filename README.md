# Intelligent Onboarding Assistant

<div align="center">

**An enterprise-grade RAG system that transforms company onboarding using GitLab's public knowledge base**

**[Live Demo](https://onboarding-ui-p5rleegxya-uc.a.run.app/)**

</div>

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
4. Name it: `mlops-storage-uploader`
5. Grant roles: "Storage Admin" and "Storage Object Admin"
6. Click "Create Key" → Choose "JSON"
7. Save the JSON file as `mlops-476419-2c1937dab204.json` in the project root

**4.2: Set Up GCS Bucket**

```bash
# Install Google Cloud SDK if not already installed
# Visit: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Create bucket (if not exists)
gsutil mb -p mlops-476419 -l us-central1 gs://mlops-data-oa

# Verify bucket
gsutil ls gs://mlops-data-oa
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
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Google Cloud Storage Configuration
GCS_BUCKET_NAME=your_gcs_bucket_name
GCS_PROJECT_ID=your_gcp_project_id

# Google Cloud Credentials (use absolute paths)
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\project\your-service-account-key.json
GCS_KEY_PATH=C:\path\to\your\project\your-service-account-key.json

# Optional: Model Configuration
EMBEDDING_MODEL=all-mpnet-base-v2
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

**Important Notes:**
- Replace `your_groq_api_key_here` with your actual Groq API key from Step 5
- Replace `your_gcs_bucket_name` with your GCS bucket name
- Replace `your_gcp_project_id` with your Google Cloud project ID
- For `GOOGLE_APPLICATION_CREDENTIALS` and `GCS_KEY_PATH`, use the **absolute path** to your service account JSON file
  - **Windows example**: `C:\Users\YourName\Desktop\project\mlops-service-account.json`
  - **macOS/Linux example**: `/home/username/project/mlops-service-account.json`
- Never commit the `.env` file or service account JSON to Git (they're in .gitignore)