"""
Logging Configuration Module
"""
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

class PipelineLogger:
    """Custom logger for the data pipeline"""
    
    def __init__(self, name: str = "pipeline", log_dir: Optional[Path] = None):
        self.name = name
        if log_dir is None:
            log_dir = Path("./data-pipeline/logs/pipeline")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up standard Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        
        # File handler
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_format)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def get_logger(self):
        """Get logger instance"""
        return self.logger

# Global logger
pipeline_logger = PipelineLogger().get_logger()