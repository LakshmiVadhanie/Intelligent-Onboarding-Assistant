import os
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent.absolute()

# Add the project root and data-pipeline to Python path
sys.path.append(str(project_root))
sys.path.append(str(project_root / "data-pipeline"))