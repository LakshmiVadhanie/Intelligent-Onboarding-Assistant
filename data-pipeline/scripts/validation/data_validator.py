import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime
import logging
from scripts.utils.logging_setup import setup_logger

logger = setup_logger('data_validator')

class DataValidator:
    def __init__(self, schema_path: Optional[str] = None):
        """Initialize DataValidator with optional schema"""
        self.schema = self._load_schema(schema_path) if schema_path else {}
        self.stats_history = []
        
    def _load_schema(self, schema_path: str) -> Dict:
        """Load schema from JSON file"""
        with open(schema_path, 'r') as f:
            return json.load(f)
    
    def generate_schema(self, data: Dict) -> Dict:
        """Generate schema from sample data"""
        schema = {
            "type": "object",
            "required": ["title", "video_id", "segments"],
            "properties": {
                "title": {"type": "string"},
                "video_id": {"type": "string"},
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "start", "end", "text"],
                        "properties": {
                            "id": {"type": "integer"},
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "text": {"type": "string"}
                        }
                    }
                }
            }
        }
        return schema
    
    def validate_schema(self, data: Dict) -> List[str]:
        """Validate data against schema"""
        errors = []
        
        # Check required fields
        for field in self.schema.get("required", []):
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate segments if present
        if "segments" in data:
            for i, segment in enumerate(data["segments"]):
                # Check segment structure
                for field in ["id", "start", "end", "text"]:
                    if field not in segment:
                        errors.append(f"Segment {i} missing field: {field}")
                
                # Check temporal consistency
                if "start" in segment and "end" in segment:
                    if segment["start"] >= segment["end"]:
                        errors.append(f"Segment {i} has invalid time range")
        
        return errors
    
    def generate_statistics(self, data: Dict) -> Dict:
        """Generate statistics from data"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_segments": len(data.get("segments", [])),
            "total_words": sum(len(seg["text"].split()) 
                             for seg in data.get("segments", [])),
            "avg_segment_duration": np.mean([
                seg["end"] - seg["start"] 
                for seg in data.get("segments", [])
            ]) if data.get("segments") else 0,
            "missing_values": sum(
                1 for seg in data.get("segments", [])
                if not seg.get("text", "").strip()
            )
        }
        
        self.stats_history.append(stats)
        return stats
    
    def detect_anomalies(self, stats: Dict) -> List[str]:
        """Detect anomalies in statistics"""
        anomalies = []
        
        if len(self.stats_history) < 2:
            return anomalies
            
        # Get previous stats for comparison
        prev_stats = self.stats_history[-2]
        
        # Check for significant changes
        thresholds = {
            "total_segments": 0.4,  # 40% change
            "total_words": 0.4,
            "avg_segment_duration": 0.3,
            "missing_values": 2  # Absolute increase
        }
        
        for metric, threshold in thresholds.items():
            if metric in stats and metric in prev_stats:
                curr_val = stats[metric]
                prev_val = prev_stats[metric]
                
                if prev_val != 0:
                    change = abs(curr_val - prev_val) / prev_val
                    if change > threshold:
                        anomalies.append(
                            f"Anomaly detected in {metric}: "
                            f"Changed by {change*100:.1f}%"
                        )
        
        return anomalies
    
    def save_statistics(self, stats: Dict, output_dir: str):
        """Save statistics to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stats_file = output_path / "data_statistics.json"

        # Ensure parent directory exists for stats_file
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing stats if file exists
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                all_stats = json.load(f)
        else:
            all_stats = []

        # Append new stats
        all_stats.append(stats)

        # Save updated stats
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)