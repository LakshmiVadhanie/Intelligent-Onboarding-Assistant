
import pandas as pd
import sys
from pathlib import Path
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score
from typing import List, Dict
import json

# Add the data-pipeline directory to the Python path
current_dir = Path(__file__).resolve().parent
data_pipeline_dir = current_dir.parent.parent
sys.path.insert(0, str(data_pipeline_dir))

class FairnessAnalyzer:
    def __init__(self, sensitive_features: List[str]):
        self.sensitive_features = sensitive_features

    def analyze(self, y_true, y_pred, X: pd.DataFrame) -> Dict:
        """
        Analyze fairness metrics across slices defined by sensitive features.
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            X: DataFrame with sensitive features as columns
        Returns:
            Dictionary of MetricFrame results for each sensitive feature
        """
        results = {}
        for feature in self.sensitive_features:
            if feature in X.columns:
                mf = MetricFrame(
                    metrics={
                        'accuracy': accuracy_score,
                        'selection_rate': selection_rate
                    },
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=X[feature]
                )
                results[feature] = {
                    'by_group': mf.by_group.to_dict(),
                    'overall': mf.overall.to_dict() if hasattr(mf.overall, 'to_dict') else mf.overall
                }
        return results

# Example usage (to be run after model training):
if __name__ == "__main__":
    print("=" * 60)
    print("FAIRNESS ANALYSIS TEST")
    print("=" * 60)
    
    # Example data
    df = pd.DataFrame({
        'gender': ['male', 'female', 'female', 'male', 'female', 'male', 'female'],
        'age_group': ['adult', 'adult', 'child', 'child', 'adult', 'adult', 'child']
    })
    y_true = [1, 0, 1, 0, 1, 1, 0]
    y_pred = [1, 0, 0, 0, 1, 1, 1]

    print("\nSample Data:")
    print("-" * 40)
    print(f"Total samples: {len(y_true)}")
    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"\nSensitive features:")
    print(df.to_string(index=False))

    analyzer = FairnessAnalyzer(sensitive_features=['gender', 'age_group'])
    fairness_results = analyzer.analyze(y_true, y_pred, df)
    
    print("\n\nFairness Analysis Results:")
    print("=" * 60)
    
    for feature, metrics in fairness_results.items():
        print(f"\n{feature.upper()}:")
        print("-" * 40)
        
        print("\nBy Group:")
        for metric_name, groups in metrics['by_group'].items():
            print(f"  {metric_name}:")
            for group, value in groups.items():
                print(f"    {group}: {value:.3f}")
        
        print("\nOverall:")
        if isinstance(metrics['overall'], dict):
            for metric_name, value in metrics['overall'].items():
                print(f"  {metric_name}: {value:.3f}")
        else:
            print(f"  Overall metrics: {metrics['overall']}")
    
    print("\n" + "=" * 60)
