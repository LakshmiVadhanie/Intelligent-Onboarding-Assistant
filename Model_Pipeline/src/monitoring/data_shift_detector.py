"""
Advanced Data Shift Detection using Evidently AI
Monitors input data distribution and compares with training data
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataShiftReport:
    """Data class for data shift reports"""
    timestamp: str
    report_id: str
    dataset_name: str
    total_features: int
    drifted_features: int
    drift_rate: float
    feature_drifts: List[Dict[str, Any]]
    data_quality_issues: List[Dict[str, Any]]
    recommendations: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    
    def to_dict(self):
        return asdict(self)


class EvidentlyAIDataShiftDetector:
    """Data shift detection using Evidently AI concepts"""
    
    def __init__(self,
                 reports_dir: str = "experiments/data_shift/evidently",
                 baseline_dir: str = "experiments/data_shift/baseline"):
        """
        Initialize Evidently AI Data Shift Detector
        
        Args:
            reports_dir: Directory to store reports
            baseline_dir: Directory to store baseline data
        """
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_profiles = {}
        self.shift_history = []
        
        self._load_baseline_profiles()
        self._load_shift_history()
    
    def _load_baseline_profiles(self) -> None:
        """Load baseline data profiles"""
        profile_file = self.baseline_dir / "baseline_profiles.json"
        if profile_file.exists():
            try:
                with open(profile_file, 'r') as f:
                    self.baseline_profiles = json.load(f)
                logger.info(f"Loaded {len(self.baseline_profiles)} baseline profiles")
            except Exception as e:
                logger.error(f"Error loading baseline profiles: {e}")
    
    def _load_shift_history(self) -> None:
        """Load shift detection history"""
        history_file = self.reports_dir / "shift_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.shift_history = json.load(f)
            except Exception as e:
                logger.error(f"Error loading shift history: {e}")
    
    def set_baseline(self,
                    data: Dict[str, np.ndarray],
                    dataset_name: str = "training_data") -> None:
        """
        Set baseline data profile from training data
        
        Args:
            data: Dictionary of feature_name -> values array
            dataset_name: Name of the dataset
        """
        baseline_profile = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "features": {}
        }
        
        for feature_name, values in data.items():
            if isinstance(values, np.ndarray):
                values_list = values.tolist()
            else:
                values_list = list(values)
            
            baseline_profile["features"][feature_name] = {
                "name": feature_name,
                "type": "numerical",  # Simplified for demo
                "mean": float(np.mean(values_list)) if values_list else 0,
                "std": float(np.std(values_list)) if values_list else 0,
                "min": float(np.min(values_list)) if values_list else 0,
                "max": float(np.max(values_list)) if values_list else 0,
                "median": float(np.median(values_list)) if values_list else 0,
                "percentile_25": float(np.percentile(values_list, 25)) if values_list else 0,
                "percentile_75": float(np.percentile(values_list, 75)) if values_list else 0,
                "missing_values": 0,
                "sample_count": len(values_list)
            }
        
        self.baseline_profiles[dataset_name] = baseline_profile
        self._save_baseline_profiles()
        logger.info(f"Baseline profile set for {dataset_name} with {len(data)} features")
    
    def _save_baseline_profiles(self) -> None:
        """Save baseline profiles"""
        try:
            profile_file = self.baseline_dir / "baseline_profiles.json"
            with open(profile_file, 'w') as f:
                json.dump(self.baseline_profiles, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving baseline profiles: {e}")
    
    def detect_data_drift(self,
                         current_data: Dict[str, np.ndarray],
                         baseline_name: str = "training_data",
                         threshold: float = 0.1) -> Optional[DataShiftReport]:
        """
        Detect data drift between current and baseline data
        
        Args:
            current_data: Dictionary of feature_name -> values
            baseline_name: Name of baseline to compare against
            threshold: Drift detection threshold (0-1)
            
        Returns:
            DataShiftReport or None if baseline not found
        """
        if baseline_name not in self.baseline_profiles:
            logger.warning(f"Baseline '{baseline_name}' not found")
            return None
        
        baseline = self.baseline_profiles[baseline_name]
        feature_drifts = []
        drifted_count = 0
        
        for feature_name, current_values in current_data.items():
            if feature_name not in baseline["features"]:
                continue
            
            baseline_stats = baseline["features"][feature_name]
            
            # Calculate current statistics
            if isinstance(current_values, np.ndarray):
                values_list = current_values.tolist()
            else:
                values_list = list(current_values)
            
            current_stats = {
                "mean": float(np.mean(values_list)) if values_list else 0,
                "std": float(np.std(values_list)) if values_list else 0,
                "min": float(np.min(values_list)) if values_list else 0,
                "max": float(np.max(values_list)) if values_list else 0,
            }
            
            # Detect drift using statistical distance
            drift_score = self._calculate_drift_score(baseline_stats, current_stats)
            is_drifted = drift_score > threshold
            
            if is_drifted:
                drifted_count += 1
            
            feature_drifts.append({
                "feature_name": feature_name,
                "drift_score": round(drift_score, 4),
                "is_drifted": is_drifted,
                "baseline_mean": baseline_stats["mean"],
                "current_mean": current_stats["mean"],
                "baseline_std": baseline_stats["std"],
                "current_std": current_stats["std"],
                "mean_change_percent": round(
                    abs(current_stats["mean"] - baseline_stats["mean"]) / 
                    (abs(baseline_stats["mean"]) + 1e-8) * 100, 2
                )
            })
        
        # Calculate overall metrics
        total_features = len(feature_drifts)
        drift_rate = drifted_count / total_features if total_features > 0 else 0
        
        # Determine severity
        if drift_rate >= 0.5:
            severity = "critical"
        elif drift_rate >= 0.3:
            severity = "high"
        elif drift_rate >= 0.1:
            severity = "medium"
        else:
            severity = "low"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(feature_drifts, severity)
        
        # Detect data quality issues
        data_quality_issues = self._detect_data_quality_issues(current_data)
        
        report = DataShiftReport(
            timestamp=datetime.now().isoformat(),
            report_id=f"shift_{datetime.now().timestamp()}",
            dataset_name=baseline_name,
            total_features=total_features,
            drifted_features=drifted_count,
            drift_rate=round(drift_rate, 4),
            feature_drifts=feature_drifts,
            data_quality_issues=data_quality_issues,
            recommendations=recommendations,
            severity=severity
        )
        
        return report
    
    def _calculate_drift_score(self, baseline: Dict, current: Dict) -> float:
        """Calculate drift score between two distributions"""
        # Normalize difference
        mean_diff = abs(current["mean"] - baseline["mean"]) / (abs(baseline["mean"]) + 1e-8)
        
        # Consider std deviation changes
        std_diff = abs(current["std"] - baseline["std"]) / (abs(baseline["std"]) + 1e-8)
        
        # Combined drift score
        drift_score = (mean_diff + std_diff) / 2
        
        return min(drift_score, 1.0)  # Cap at 1.0
    
    def _detect_data_quality_issues(self,
                                   current_data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect data quality issues"""
        issues = []
        
        for feature_name, values in current_data.items():
            if isinstance(values, np.ndarray):
                values_list = values.tolist()
            else:
                values_list = list(values)
            
            # Check for missing values
            missing_count = sum(1 for v in values_list if v is None or (isinstance(v, float) and np.isnan(v)))
            if missing_count > 0:
                issues.append({
                    "feature": feature_name,
                    "issue": "missing_values",
                    "count": missing_count,
                    "percentage": round(missing_count / len(values_list) * 100, 2),
                    "severity": "high" if missing_count / len(values_list) > 0.1 else "medium"
                })
            
            # Check for outliers (values beyond 3 std)
            if len(values_list) > 0:
                mean = np.mean(values_list)
                std = np.std(values_list)
                outliers = sum(1 for v in values_list if abs(v - mean) > 3 * std)
                if outliers > 0:
                    issues.append({
                        "feature": feature_name,
                        "issue": "outliers",
                        "count": outliers,
                        "percentage": round(outliers / len(values_list) * 100, 2),
                        "severity": "medium"
                    })
        
        return issues
    
    def _generate_recommendations(self,
                                 feature_drifts: List[Dict],
                                 severity: str) -> List[str]:
        """Generate recommendations based on drift analysis"""
        recommendations = []
        
        drifted_features = [f for f in feature_drifts if f["is_drifted"]]
        
        if severity == "critical":
            recommendations.append("⚠️ CRITICAL: Significant data shift detected. Consider immediate model retraining.")
            recommendations.append(f"   - {len(drifted_features)} out of {len(feature_drifts)} features show drift")
            recommendations.append("   - Review and retrain model with recent data")
        
        elif severity == "high":
            recommendations.append("⚠️ HIGH: Notable data shift detected.")
            recommendations.append(f"   - {len(drifted_features)} features show drift")
            recommendations.append("   - Monitor closely and plan for retraining")
        
        elif severity == "medium":
            recommendations.append("ℹ️ MEDIUM: Moderate data shift detected.")
            recommendations.append(f"   - {len(drifted_features)} features affected")
            recommendations.append("   - Continue monitoring and prepare retraining pipeline")
        
        else:
            recommendations.append("✓ LOW: Minimal data shift detected.")
            recommendations.append("   - Continue monitoring")
        
        # Feature-specific recommendations
        for drift in sorted(drifted_features, key=lambda x: x["drift_score"], reverse=True)[:3]:
            change = drift["mean_change_percent"]
            recommendations.append(
                f"   - Feature '{drift['feature_name']}' changed by {change}% "
                f"(drift score: {drift['drift_score']})"
            )
        
        return recommendations
    
    def save_report(self, report: DataShiftReport) -> None:
        """Save data shift report"""
        self.shift_history.append(report.to_dict())
        
        try:
            # Save to history
            history_file = self.reports_dir / "shift_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.shift_history, f, indent=2, default=str)
            
            # Save individual report
            report_file = self.reports_dir / f"{report.report_id}.json"
            with open(report_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Report saved: {report.report_id}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def get_shift_summary(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get summary of data shifts"""
        history = self.shift_history.copy()
        if limit:
            history = history[-limit:]
        
        total = len(history)
        critical_count = len([h for h in history if h.get("severity") == "critical"])
        high_count = len([h for h in history if h.get("severity") == "high"])
        
        return {
            "total_reports": total,
            "critical_shifts": critical_count,
            "high_shifts": high_count,
            "recent_reports": history[-5:] if history else [],
            "avg_drift_rate": round(np.mean([h.get("drift_rate", 0) for h in history]), 4) if history else 0
        }
    
    def export_report(self, export_path: str, format: str = "json") -> None:
        """Export reports"""
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "json":
                with open(export_path, 'w') as f:
                    json.dump(self.shift_history, f, indent=2, default=str)
            elif format == "csv":
                import csv
                if self.shift_history:
                    with open(export_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=self.shift_history[0].keys())
                        writer.writeheader()
                        writer.writerows(self.shift_history)
            logger.info(f"Report exported to {export_path}")
        except Exception as e:
            logger.error(f"Error exporting report: {e}")


class TFDVDataValidation:
    """TensorFlow Data Validation (TFDV) concepts implementation"""
    
    def __init__(self,
                 validation_dir: str = "experiments/data_shift/tfdv"):
        """
        Initialize TFDV Data Validation
        
        Args:
            validation_dir: Directory to store validation results
        """
        self.validation_dir = Path(validation_dir)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        self.schema = {}
        self.validation_results = []
    
    def infer_schema(self,
                    data: Dict[str, np.ndarray],
                    dataset_name: str = "default") -> Dict[str, Any]:
        """
        Infer schema from data
        
        Args:
            data: Dictionary of feature_name -> values
            dataset_name: Name of dataset
            
        Returns:
            Schema dictionary
        """
        schema = {
            "name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "features": {}
        }
        
        for feature_name, values in data.items():
            if isinstance(values, np.ndarray):
                values_list = values.tolist()
            else:
                values_list = list(values)
            
            # Infer type
            if all(isinstance(v, (int, np.integer)) for v in values_list if v is not None):
                feature_type = "INT"
            elif all(isinstance(v, (float, np.floating)) for v in values_list if v is not None):
                feature_type = "FLOAT"
            else:
                feature_type = "STRING"
            
            schema["features"][feature_name] = {
                "name": feature_name,
                "type": feature_type,
                "presence": "REQUIRED",
                "valency": "SINGLE",
                "value_count": {
                    "min": 1,
                    "max": 1
                },
                "domain": f"{feature_name}_domain",
                "shape": {
                    "dim": [
                        {
                            "size": len(values_list)
                        }
                    ]
                }
            }
        
        self.schema = schema
        return schema
    
    def validate_data(self,
                     data: Dict[str, np.ndarray],
                     schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate data against schema
        
        Args:
            data: Dictionary of feature_name -> values
            schema: Schema to validate against
            
        Returns:
            Validation results
        """
        if schema is None:
            schema = self.schema
        
        if not schema:
            logger.warning("No schema available for validation")
            return {"valid": True, "errors": []}
        
        validation_result = {
            "timestamp": datetime.now().isoformat(),
            "valid": True,
            "errors": [],
            "anomalies": []
        }
        
        for feature_name, feature_schema in schema.get("features", {}).items():
            if feature_name not in data:
                validation_result["errors"].append({
                    "feature": feature_name,
                    "error": "MISSING_FEATURE",
                    "message": f"Feature '{feature_name}' not found in data"
                })
                validation_result["valid"] = False
                continue
            
            values = data[feature_name]
            if isinstance(values, np.ndarray):
                values_list = values.tolist()
            else:
                values_list = list(values)
            
            # Validate type
            expected_type = feature_schema.get("type", "FLOAT")
            for v in values_list:
                if v is not None:
                    if expected_type == "INT" and not isinstance(v, (int, np.integer)):
                        validation_result["errors"].append({
                            "feature": feature_name,
                            "error": "TYPE_MISMATCH",
                            "message": f"Expected {expected_type}, got {type(v).__name__}"
                        })
                        validation_result["valid"] = False
                        break
        
        return validation_result
    
    def detect_anomalies(self,
                        data: Dict[str, np.ndarray],
                        threshold: float = 0.95) -> List[Dict[str, Any]]:
        """
        Detect anomalies in data
        
        Args:
            data: Dictionary of feature_name -> values
            threshold: Anomaly detection threshold
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        for feature_name, values in data.items():
            if isinstance(values, np.ndarray):
                values_list = values.tolist()
            else:
                values_list = list(values)
            
            if not values_list:
                continue
            
            # Statistical anomalies
            mean = np.mean(values_list)
            std = np.std(values_list)
            
            extreme_values = [
                (i, v) for i, v in enumerate(values_list)
                if abs(v - mean) > 3 * std
            ]
            
            if extreme_values:
                anomalies.append({
                    "feature": feature_name,
                    "anomaly_type": "STATISTICAL_OUTLIER",
                    "count": len(extreme_values),
                    "percentage": round(len(extreme_values) / len(values_list) * 100, 2),
                    "examples": [v for _, v in extreme_values[:3]]
                })
        
        return anomalies
    
    def save_validation_result(self, result: Dict[str, Any]) -> None:
        """Save validation result"""
        self.validation_results.append(result)
        
        try:
            results_file = self.validation_dir / "validation_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            logger.info("Validation result saved")
        except Exception as e:
            logger.error(f"Error saving validation result: {e}")
