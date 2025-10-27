# dags/scripts/logging_utils.py
import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name: str):
    """
    Creates a logger that logs to both console (INFO) and a rotating file.
    File path defaults to /opt/airflow/logs/pipeline.log when running in Docker,
    and falls back to ./logs/pipeline.log when running locally.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (avoid duplicate handlers on re-import)

    logger.setLevel(logging.INFO)

    # Decide log file path (works local & in Airflow)
    log_path = os.getenv("PIPELINE_LOG_PATH", "/opt/airflow/logs/pipeline.log")
    if not os.path.isdir(os.path.dirname(log_path)):
        # Local fallback: ./logs/pipeline.log
        os.makedirs("./logs", exist_ok=True)
        log_path = "./logs/pipeline.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file (5 MB x 3 files)
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
