# Dockerfile for MLOps Onboarding Assistant
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIRFLOW_HOME=/app/data-pipeline/airflow \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__DAGS_FOLDER=/app/data-pipeline/airflow/dags \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    wget \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY data-pipeline/requirements.txt /app/data-pipeline/requirements.txt

# Install Python dependencies
# Install Airflow first, then other packages (excluding conflicting ones)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir apache-airflow[postgres]==2.10.2 psycopg2-binary && \
    grep -v "apache-airflow\|psycopg2-binary\|pydantic\|evidently\|great-expectations" /app/data-pipeline/requirements.txt | \
    grep -v "^#" | grep -v "^$" | xargs -r pip install --no-cache-dir || true && \
    pip install --no-cache-dir pydantic==2.5.0

# Copy project files
COPY . /app/

# Create necessary directories
RUN mkdir -p \
    /app/data-pipeline/data/raw/blogs \
    /app/data-pipeline/data/raw/handbook \
    /app/data-pipeline/data/raw/transcripts \
    /app/data-pipeline/data/processed/cleaned \
    /app/data-pipeline/data/processed/chunked \
    /app/data-pipeline/data/curated/final \
    /app/data-pipeline/data/cache \
    /app/data-pipeline/logs/pipeline \
    /app/data-pipeline/logs/errors \
    /app/data-pipeline/logs/airflow \
    /app/data-pipeline/airflow/logs \
    /app/data-pipeline/airflow/dags

# Set permissions
RUN chmod -R 755 /app/data-pipeline

# Expose Airflow webserver port
EXPOSE 8080

# Default command
CMD ["bash"]
