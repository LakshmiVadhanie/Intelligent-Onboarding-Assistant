# ui/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# copy only requirements and install (to allow caching)
COPY ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy UI app
COPY ui /app/ui
# If your UI is in Model_Pipeline/ui, change above accordingly:
# COPY Model_Pipeline/ui /app/ui

# Cloud Run uses PORT env var. Tell streamlit to use it.
ENV PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=${PORT}
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Start Streamlit in a way that uses PORT env var.
CMD ["sh", "-c", "streamlit run /app/ui/app.py --server.port $PORT --server.headless true"]
