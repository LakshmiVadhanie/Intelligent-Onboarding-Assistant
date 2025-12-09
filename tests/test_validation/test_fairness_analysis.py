import pandas as pd
import pytest
from scripts.validation.fairness_analysis import FairnessAnalyzer

def test_fairness_analyzer_basic():
    df = pd.DataFrame({
        'gender': ['male', 'female', 'female', 'male', 'female'],
        'age_group': ['adult', 'adult', 'child', 'child', 'adult']
    })
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 0, 0, 1]

    analyzer = FairnessAnalyzer(sensitive_features=['gender', 'age_group'])
    results = analyzer.analyze(y_true, y_pred, df)

    assert 'gender' in results
    assert 'age_group' in results
    # Check that by_group results exist and are dicts
    assert isinstance(results['gender']['by_group'], dict)
    assert isinstance(results['age_group']['by_group'], dict)
    # Check that overall metrics exist
    assert 'accuracy' in results['gender']['overall']
    assert 'selection_rate' in results['age_group']['overall']
