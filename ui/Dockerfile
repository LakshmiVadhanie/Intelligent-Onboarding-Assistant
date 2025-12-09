FROM python:3.11-slim
WORKDIR /app

# install build deps required by some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use the Model_Pipeline requirements (option B) - ensure this file exists in repo
COPY Model_Pipeline/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy code: Model_Pipeline (actual app) and ui wrapper
COPY Model_Pipeline /app/Model_Pipeline
COPY ui /app/ui

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false

CMD ["sh", "-c", "streamlit run /app/ui/app.py --server.port ${PORT:-8080} --server.address 0.0.0.0"]
