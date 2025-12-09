# ui/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install build tools used by some wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (cache)
COPY ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy UI code
COPY ui /app/ui
WORKDIR /app/ui

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false

EXPOSE 8080

# Run streamlit on port from $PORT else 8080
CMD ["sh", "-c", "streamlit run /app/ui/app.py --server.port ${PORT:-8080} --server.address 0.0.0.0"]
