#!/bin/bash

# Fix Airflow Setup Script
echo "Fixing Airflow configuration..."

# Get absolute path
ABSOLUTE_PATH=$(pwd)

# Set Airflow home to absolute path
export AIRFLOW_HOME="$ABSOLUTE_PATH/data-pipeline/airflow"

# Create Airflow directories
mkdir -p $AIRFLOW_HOME
mkdir -p $AIRFLOW_HOME/dags
mkdir -p $AIRFLOW_HOME/logs
mkdir -p $AIRFLOW_HOME/plugins

# Create airflow.cfg with absolute path
cat > $AIRFLOW_HOME/airflow.cfg << EOF
[core]
dags_folder = $AIRFLOW_HOME/dags
executor = SequentialExecutor
load_examples = False

[database]
sql_alchemy_conn = sqlite:///$AIRFLOW_HOME/airflow.db

[webserver]
web_server_port = 8080
EOF

echo "Airflow configuration fixed!"
echo "AIRFLOW_HOME set to: $AIRFLOW_HOME"
echo ""
echo "To initialize Airflow, run:"
echo "export AIRFLOW_HOME=$AIRFLOW_HOME"
echo "airflow db init"