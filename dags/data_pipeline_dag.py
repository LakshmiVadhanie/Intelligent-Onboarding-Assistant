from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# Define default args
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
}

# Create DAG
with DAG(
    dag_id='data_pipeline_dag',
    default_args=default_args,
    description='GitLab Handbook + YouTube Summarization Data Pipeline',
    schedule_interval=None,  # manual trigger
    catchup=False,
    tags=['onboarding-ai'],
) as dag:

    # Scrape GitLab Handbook
    scrape_task = BashOperator(
        task_id='scrape_handbook',
        bash_command='python /opt/airflow/dags/scripts/scraper.py',
    )

    # Transcribe YouTube Playlist
    VIDEO_LIMIT = 1

    transcribe_task = BashOperator(
    task_id='transcribe_youtube',
    bash_command=f'python /opt/airflow/dags/scripts/transcription.py --limit {VIDEO_LIMIT}',
    )

    # Validate Combined Outputs
    validate_task = BashOperator(
        task_id='validate_data',
        bash_command='python /opt/airflow/dags/scripts/validate_data.py',
    )

    # Define execution order
    scrape_task >> transcribe_task >> validate_task
