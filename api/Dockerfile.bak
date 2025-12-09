# api/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY Model_Pipeline/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY Model_Pipeline /app

ENV PYTHONUNBUFFERED=1

# Example: expose port 8080 (Cloud Run default)
EXPOSE 8080
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8080", "--workers", "1"]
