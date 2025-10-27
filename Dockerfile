FROM apache/airflow:2.7.2

USER root
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean
USER airflow

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
