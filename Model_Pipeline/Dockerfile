FROM python:3.13
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libre2-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY experiments/ ./experiments/
COPY test_pipeline.py .

RUN mkdir -p models/embeddings models/vector_store models/registry experiments/mlruns

EXPOSE 8000

ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["python", "test_pipeline.py"]