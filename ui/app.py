# ui/app.py - small wrapper to reuse Model_Pipeline/app.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Model_Pipeline"))

# now import the existing Streamlit app module
# ensure Model_Pipeline/app.py exposes a main() or defines streamlit UI at import
import app as model_app  # Model_Pipeline/app.py
# nothing else needed; streamlit will run the app code in app.py
