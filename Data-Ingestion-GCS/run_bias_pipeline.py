"""
Run Bias Detection → Mitigation Pipeline
========================================
Combines both steps for seamless debiasing.
"""

from bias_detection import run_bias_detection
from bias_mitigation import run_bias_mitigation
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bias_pipeline")

if __name__ == "__main__":
    logger.info("Starting Bias Detection + Mitigation Pipeline...")
    report_path = run_bias_detection()
    logger.info(f"Bias report generated: {report_path}")

    output_dir = run_bias_mitigation()
    logger.info(f"✨ Debiased data ready in: {output_dir}")
