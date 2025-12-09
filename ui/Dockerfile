FROM python:3.11-slim

WORKDIR /app

# small build tools for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app and code into image (we created ui/ with Model_Pipeline files)
COPY ui /app

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080
CMD ["sh", "-c", "streamlit run /app/app.py --server.port ${PORT:-8080} --server.address 0.0.0.0"]
