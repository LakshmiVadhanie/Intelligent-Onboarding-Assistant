# ui/Dockerfile - run Model_Pipeline/app.py directly (no wrapper)
FROM python:3.11-slim

WORKDIR /app

# Install system deps used by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the whole repo into the image so Model_Pipeline and other modules are available
COPY . /app

# Put repo and Model_Pipeline on PYTHONPATH so internal imports resolve
ENV PYTHONPATH=/app:/app/Model_Pipeline

# Filter Model_Pipeline/requirements.txt to remove problematic lines, then install
# (adjust the sed filter if your requirements file has other invalid tokens)
RUN if [ -f /app/Model_Pipeline/requirements.txt ]; then \
      sed -E '/^\s*$/d; /^\s*__future__/d; /^\s*airflow\b/I d; /^\s*dags\b/I d; /^\s*src\b/I d; /^\s*Model_Pipeline\b/I d; /^\s*__main__\b/I d' \
        /app/Model_Pipeline/requirements.txt > /tmp/filtered-requirements.txt && \
      echo "---- filtered requirements (head) ----" && sed -n '1,200p' /tmp/filtered-requirements.txt && \
      pip install --no-cache-dir -r /tmp/filtered-requirements.txt ; \
    else \
      echo "No Model_Pipeline/requirements.txt found; skipping pip install"; \
    fi

# Ensure Streamlit runs headless for Cloud Run
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV PYTHONUNBUFFERED=1

# Streamlit expects port from PORT env var (Cloud Run sets PORT)
EXPOSE 8080

# Start Streamlit pointing to the app inside Model_Pipeline
CMD ["sh", "-c", "streamlit run /app/Model_Pipeline/app.py --server.port ${PORT:-8080} --server.address 0.0.0.0"]
