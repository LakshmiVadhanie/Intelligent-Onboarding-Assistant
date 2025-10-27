from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from datetime import datetime

# Define default args
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['zankhujunk@gmail.com'],       
    'email_on_failure': True,                
    'email_on_retry': False,
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

    # Bias Detection Task
    bias_task = BashOperator(
    task_id='bias_detection',
    bash_command='python /opt/airflow/dags/scripts/bias_detection.py',
    )

    # Bias Mitigation Task
    bias_mitigation_task = BashOperator(
    task_id='bias_mitigation',
    bash_command='python /opt/airflow/dags/scripts/bias_mitigation.py',
    )

    # Notifying mail on success
    notify_success = EmailOperator(
    task_id='notify_success',
    to='zankhujunk@gmail.com',
    subject='Data Pipeline Succeeded',
    html_content="""
        <h3>Intelligent Onboarding Data Pipeline Completed Successfully!</h3>
        <p>Both GitLab and YouTube data have been processed and validated.</p>
    """,
    )

    # Define execution order
    scrape_task >> transcribe_task >> validate_task >> bias_task >> bias_mitigation_task >> notify_success
