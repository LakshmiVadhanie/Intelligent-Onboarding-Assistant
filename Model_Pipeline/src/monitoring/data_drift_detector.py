"""
Data Drift Detector for Monitoring Input Data Distribution
Detects shifts in data distribution using statistical methods
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats
import pickle
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftDetectionResult:
    """Data class for drift detection results"""
    timestamp: str
    metric_name: str
    is_drift_detected: bool
    drift_score: float
    p_value: float
    threshold: float
    baseline_mean: float
    baseline_std: float
    current_mean: float
    current_std: float
    samples_count: int
    drift_type: str  # 'statistical', 'distribution', 'feature'
    
    def to_dict(self):
        return asdict(self)


class DataDriftDetector:
    """Detects data drift using statistical and distribution methods"""
    
    def __init__(self,
                 baseline_dir: str = "experiments/monitoring/baseline",
                 drift_reports_dir: str = "experiments/monitoring/drift_reports",
                 statistical_threshold: float = 0.05,
                 ks_threshold: float = 0.05,
                 embedding_threshold: float = 0.8):
        """
        Initialize DataDriftDetector
        
        Args:
            baseline_dir: Directory to store baseline distributions
            drift_reports_dir: Directory to store drift reports
            statistical_threshold: P-value threshold for statistical tests (alpha)
            ks_threshold: Threshold for Kolmogorov-Smirnov test
            embedding_threshold: Cosine similarity threshold for embeddings
        """
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        self.drift_reports_dir = Path(drift_reports_dir)
        self.drift_reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.statistical_threshold = statistical_threshold
        self.ks_threshold = ks_threshold
        self.embedding_threshold = embedding_threshold
        
        self.baseline_stats = self._load_baseline_stats()
        self.drift_history = self._load_drift_history()
    
    def _load_baseline_stats(self) -> Dict[str, Any]:
        """Load baseline statistics"""
        baseline_file = self.baseline_dir / "baseline_stats.pkl"
        if baseline_file.exists():
            try:
                with open(baseline_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading baseline stats: {e}")
        return {}
    
    def _load_drift_history(self) -> List[Dict[str, Any]]:
        """Load drift detection history"""
        history_file = self.drift_reports_dir / "drift_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading drift history: {e}")
        return []
    
    def set_baseline(self,
                    data: np.ndarray,
                    feature_name: str = "default",
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Set baseline distribution for a feature
        
        Args:
            data: Numpy array of baseline data
            feature_name: Name of the feature
            metadata: Additional metadata about the baseline
        """
        if data.size == 0:
            logger.warning(f"Empty data for baseline: {feature_name}")
            return
        
        baseline_info = {
            "feature_name": feature_name,
            "timestamp": datetime.now().isoformat(),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data)),
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "sample_count": len(data),
            "metadata": metadata or {}
        }
        
        self.baseline_stats[feature_name] = baseline_info
        
        # Save baseline
        try:
            baseline_file = self.baseline_dir / "baseline_stats.pkl"
            with open(baseline_file, 'wb') as f:
                pickle.dump(self.baseline_stats, f)
            logger.info(f"Baseline set for feature: {feature_name}")
        except Exception as e:
            logger.error(f"Error saving baseline: {e}")
    
    def detect_statistical_drift(self,
                                current_data: np.ndarray,
                                feature_name: str = "default",
                                method: str = "ks") -> Optional[DriftDetectionResult]:
        """
        Detect drift using statistical tests
        
        Args:
            current_data: Current data to test
            feature_name: Name of the feature
            method: Test method ('ks', 'ttest', 'mannwhitneyu')
            
        Returns:
            DriftDetectionResult or None if baseline not set
        """
        if feature_name not in self.baseline_stats:
            logger.warning(f"No baseline for feature: {feature_name}")
            return None
        
        baseline_info = self.baseline_stats[feature_name]
        baseline_mean = baseline_info["mean"]
        baseline_std = baseline_info["std"]
        
        current_mean = float(np.mean(current_data))
        current_std = float(np.std(current_data))
        
        # Perform statistical test
        if method == "ks":
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(
                current_data,
                np.random.normal(baseline_mean, baseline_std, len(current_data))
            )
            threshold = self.ks_threshold
        elif method == "ttest":
            # T-test
            statistic, p_value = stats.ttest_ind(
                current_data,
                np.random.normal(baseline_mean, baseline_std, len(current_data))
            )
            threshold = self.statistical_threshold
        elif method == "mannwhitneyu":
            # Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(
                current_data,
                np.random.normal(baseline_mean, baseline_std, len(current_data))
            )
            threshold = self.statistical_threshold
        else:
            raise ValueError(f"Unknown test method: {method}")
        
        is_drift = p_value < threshold
        
        result = DriftDetectionResult(
            timestamp=datetime.now().isoformat(),
            metric_name=feature_name,
            is_drift_detected=is_drift,
            drift_score=float(statistic),
            p_value=float(p_value),
            threshold=threshold,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            current_mean=current_mean,
            current_std=current_std,
            samples_count=len(current_data),
            drift_type="statistical"
        )
        
        return result
    
    def detect_distribution_drift(self,
                                 current_data: np.ndarray,
                                 feature_name: str = "default",
                                 test_method: str = "ks") -> Optional[DriftDetectionResult]:
        """
        Detect drift using distribution comparison
        
        Args:
            current_data: Current data to test
            feature_name: Name of the feature
            test_method: KS test or Wasserstein distance
            
        Returns:
            DriftDetectionResult
        """
        if feature_name not in self.baseline_stats:
            logger.warning(f"No baseline for feature: {feature_name}")
            return None
        
        baseline_info = self.baseline_stats[feature_name]
        baseline_mean = baseline_info["mean"]
        baseline_std = baseline_info["std"]
        
        current_mean = float(np.mean(current_data))
        current_std = float(np.std(current_data))
        
        # Generate synthetic baseline for comparison
        synthetic_baseline = np.random.normal(
            baseline_mean,
            baseline_std,
            len(current_data)
        )
        
        # Perform KS test
        statistic, p_value = stats.ks_2samp(current_data, synthetic_baseline)
        
        is_drift = p_value < self.ks_threshold
        
        result = DriftDetectionResult(
            timestamp=datetime.now().isoformat(),
            metric_name=feature_name,
            is_drift_detected=is_drift,
            drift_score=float(statistic),
            p_value=float(p_value),
            threshold=self.ks_threshold,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            current_mean=current_mean,
            current_std=current_std,
            samples_count=len(current_data),
            drift_type="distribution"
        )
        
        return result
    
    def detect_embedding_drift(self,
                              baseline_embeddings: np.ndarray,
                              current_embeddings: np.ndarray,
                              feature_name: str = "embeddings") -> Optional[DriftDetectionResult]:
        """
        Detect drift in embeddings using cosine similarity
        
        Args:
            baseline_embeddings: Baseline embedding vectors
            current_embeddings: Current embedding vectors
            feature_name: Name of the feature
            
        Returns:
            DriftDetectionResult
        """
        # Normalize embeddings
        baseline_norm = baseline_embeddings / np.linalg.norm(baseline_embeddings, axis=1, keepdims=True)
        current_norm = current_embeddings / np.linalg.norm(current_embeddings, axis=1, keepdims=True)
        
        # Compute centroid cosine similarity
        baseline_centroid = np.mean(baseline_norm, axis=0)
        current_centroid = np.mean(current_norm, axis=0)
        
        similarity = float(np.dot(baseline_centroid, current_centroid))
        
        # Higher similarity = less drift
        drift_score = 1.0 - similarity
        is_drift = similarity < self.embedding_threshold
        
        # Estimate p-value based on cosine distance
        distances = 1 - np.dot(baseline_norm, current_norm.T).diagonal()
        p_value = float(np.mean(distances))
        
        result = DriftDetectionResult(
            timestamp=datetime.now().isoformat(),
            metric_name=feature_name,
            is_drift_detected=is_drift,
            drift_score=drift_score,
            p_value=p_value,
            threshold=1.0 - self.embedding_threshold,
            baseline_mean=baseline_centroid.mean(),
            baseline_std=baseline_centroid.std(),
            current_mean=current_centroid.mean(),
            current_std=current_centroid.std(),
            samples_count=len(current_embeddings),
            drift_type="feature"
        )
        
        return result
    
    def save_drift_report(self, result: DriftDetectionResult) -> None:
        """Save drift detection result"""
        self.drift_history.append(result.to_dict())
        
        try:
            history_file = self.drift_reports_dir / "drift_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.drift_history, f, indent=2, default=str)
            
            # Save individual report
            report_file = self.drift_reports_dir / f"{result.timestamp.replace(':', '-')}.json"
            with open(report_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Drift report saved for {result.metric_name}")
        except Exception as e:
            logger.error(f"Error saving drift report: {e}")
    
    def get_drift_summary(self,
                         limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Get summary of detected drifts
        
        Args:
            limit: Limit to recent N records
            
        Returns:
            Dictionary with drift summary
        """
        history = self.drift_history.copy()
        
        if limit:
            history = history[-limit:]
        
        total = len(history)
        drifts = [h for h in history if h.get("is_drift_detected", False)]
        
        return {
            "total_checks": total,
            "drift_count": len(drifts),
            "drift_rate": len(drifts) / total if total > 0 else 0,
            "recent_drifts": drifts[-5:] if drifts else [],
            "by_feature": self._group_drifts_by_feature(history)
        }
    
    def _group_drifts_by_feature(self, history: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group drift counts by feature"""
        grouped = {}
        for record in history:
            feature = record.get("metric_name", "unknown")
            grouped[feature] = grouped.get(feature, 0)
            if record.get("is_drift_detected", False):
                grouped[feature] += 1
        return grouped
    
    def export_drift_report(self, export_path: str, format: str = "json") -> None:
        """Export drift reports"""
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "json":
                with open(export_path, 'w') as f:
                    json.dump(self.drift_history, f, indent=2, default=str)
            elif format == "csv":
                import pandas as pd
                df = pd.DataFrame(self.drift_history)
                df.to_csv(export_path, index=False)
            logger.info(f"Drift report exported to {export_path}")
        except Exception as e:
            logger.error(f"Error exporting drift report: {e}")
