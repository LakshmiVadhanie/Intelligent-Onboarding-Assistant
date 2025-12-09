# ui/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY ui/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY ui /app
EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.headless", "true"]
