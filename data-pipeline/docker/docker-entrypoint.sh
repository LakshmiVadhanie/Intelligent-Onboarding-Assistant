#!/bin/bash
set -e

# Docker entrypoint script for MLOps Onboarding Assistant

echo "================================================"
echo "MLOps Onboarding Assistant - Docker Setup"
echo "================================================"

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h postgres -U airflow; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 2
done
echo "PostgreSQL is up!"

# Initialize Airflow database
if [ "$1" = "webserver" ]; then
    echo "Initializing Airflow database..."
    airflow db migrate
    
    echo "Creating Airflow admin user..."
    airflow users create \
        --username "${AIRFLOW_ADMIN_USERNAME:-admin}" \
        --password "${AIRFLOW_ADMIN_PASSWORD:-admin}" \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email "${AIRFLOW_ADMIN_EMAIL:-admin@example.com}" || echo "User already exists"
    
    echo "Starting Airflow webserver..."
    exec airflow webserver --port 8080
    
elif [ "$1" = "scheduler" ]; then
    echo "Starting Airflow scheduler..."
    exec airflow scheduler
    
elif [ "$1" = "worker" ]; then
    echo "Starting Airflow worker..."
    exec airflow celery worker
    
else
    echo "Starting bash shell..."
    exec bash
fi
