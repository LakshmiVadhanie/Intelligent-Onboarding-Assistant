# ui/app.py - wrapper that delegates to Model_Pipeline/app.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Model_Pipeline"))

# Import the streamlit app file from Model_Pipeline (it should create the UI at import)
import app  # Model_Pipeline/app.py
