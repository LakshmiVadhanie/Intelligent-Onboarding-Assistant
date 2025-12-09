import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_dir="data-pipeline/logs"):
    """Set up a logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler - daily rotation
    today = datetime.now().strftime('%Y-%m-%d')
    file_handler = logging.FileHandler(
        log_dir / f"{name}_{today}.log"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def log_pipeline_step(logger, step_name, status, details=None):
    """Log a pipeline step with standardized formatting"""
    msg = f"Pipeline Step: {step_name} - Status: {status}"
    if details:
        msg += f" - Details: {details}"
    
    if status == "SUCCESS":
        logger.info(msg)
    elif status == "WARNING":
        logger.warning(msg)
    elif status == "ERROR":
        logger.error(msg)
    else:
        logger.info(msg)

def log_data_stats(logger, stats_dict):
    """Log data statistics in a structured format"""
    logger.info("Data Statistics:")
    for key, value in stats_dict.items():
        logger.info(f"  {key}: {value}")