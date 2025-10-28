
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score
from typing import List, Dict

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
    # Example data
    df = pd.DataFrame({
        'gender': ['male', 'female', 'female', 'male', 'female'],
        'age_group': ['adult', 'adult', 'child', 'child', 'adult']
    })
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 0, 0, 1]

    analyzer = FairnessAnalyzer(sensitive_features=['gender', 'age_group'])
    fairness_results = analyzer.analyze(y_true, y_pred, df)
    print(fairness_results)
