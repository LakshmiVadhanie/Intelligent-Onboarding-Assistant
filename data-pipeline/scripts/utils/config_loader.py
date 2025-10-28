"""
Configuration Loader Module
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConfigLoader:
    """Loads and manages pipeline configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Navigate from utils folder to configs folder
            config_path = Path(__file__).parent.parent.parent / "configs" / "pipeline_config.yaml"
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._substitute_env_vars()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _substitute_env_vars(self) -> None:
        """Substitute environment variables"""
        def substitute(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and v.startswith('${'):
                        var_name = v[2:-1]
                        obj[k] = os.getenv(var_name, v)
                    elif isinstance(v, (dict, list)):
                        substitute(v)
            elif isinstance(obj, list):
                for item in obj:
                    substitute(item)
        substitute(self.config)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value by dot-notation path"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

# Global instance
config = ConfigLoader()