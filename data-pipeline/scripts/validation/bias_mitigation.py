import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging

# Add the data-pipeline directory to the Python path
current_dir = Path(__file__).resolve().parent
data_pipeline_dir = current_dir.parent.parent
sys.path.insert(0, str(data_pipeline_dir))

from scripts.utils.logging_setup import setup_logger

logger = setup_logger('bias_mitigation')


class BiasMitigation:
    """
    Bias mitigation strategies including oversampling techniques
    to balance datasets across sensitive attributes.
    """
    
    def __init__(self, sensitive_features: List[str]):
        """
        Initialize BiasMitigation with sensitive features to monitor.
        
        Args:
            sensitive_features: List of column names representing sensitive attributes
        """
        self.sensitive_features = sensitive_features
        logger.info(f"Initialized BiasMitigation with features: {sensitive_features}")
    
    def analyze_distribution(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze the distribution of sensitive features in the dataset.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            Dictionary with distribution statistics for each sensitive feature
        """
        distribution = {}
        
        for feature in self.sensitive_features:
            if feature in df.columns:
                counts = df[feature].value_counts()
                total = len(df)
                
                distribution[feature] = {
                    'counts': counts.to_dict(),
                    'percentages': (counts / total * 100).to_dict(),
                    'total': total,
                    'unique_values': len(counts),
                    'imbalance_ratio': counts.max() / counts.min() if counts.min() > 0 else float('inf')
                }
                
                logger.info(f"Feature '{feature}' distribution: {distribution[feature]['counts']}")
        
        return distribution
    
    def random_oversampling(self, df: pd.DataFrame, target_column: str = None,
                           strategy: str = 'auto') -> pd.DataFrame:
        """
        Apply random oversampling to balance the dataset.
        
        Args:
            df: DataFrame to oversample
            target_column: Target column to balance. If None, balance sensitive features
            strategy: 'auto' (balance to majority), 'equal' (equal distribution), 
                     or int (target count per class)
            
        Returns:
            Oversampled DataFrame
        """
        logger.info(f"Applying random oversampling with strategy: {strategy}")
        
        if target_column and target_column in df.columns:
            return self._oversample_by_column(df, target_column, strategy)
        else:
            # Oversample based on sensitive features
            return self._oversample_sensitive_features(df, strategy)
    
    def _oversample_by_column(self, df: pd.DataFrame, column: str, 
                             strategy: str) -> pd.DataFrame:
        """Oversample based on a specific column."""
        value_counts = df[column].value_counts()
        
        if strategy == 'auto':
            target_count = value_counts.max()
        elif strategy == 'equal':
            target_count = int(len(df) / len(value_counts))
        elif isinstance(strategy, int):
            target_count = strategy
        else:
            logger.warning(f"Unknown strategy: {strategy}, using 'auto'")
            target_count = value_counts.max()
        
        logger.info(f"Target count per class: {target_count}")
        
        # Separate majority and minority classes
        oversampled_dfs = []
        
        for value in value_counts.index:
            class_df = df[df[column] == value]
            current_count = len(class_df)
            
            if current_count < target_count:
                # Oversample minority class
                n_samples = target_count - current_count
                oversampled = class_df.sample(n=n_samples, replace=True, random_state=42)
                oversampled_dfs.append(pd.concat([class_df, oversampled], ignore_index=True))
                logger.info(f"Oversampled class '{value}': {current_count} -> {target_count}")
            else:
                oversampled_dfs.append(class_df)
        
        result = pd.concat(oversampled_dfs, ignore_index=True)
        logger.info(f"Original size: {len(df)}, Oversampled size: {len(result)}")
        
        return result
    
    def _oversample_sensitive_features(self, df: pd.DataFrame, 
                                      strategy: str) -> pd.DataFrame:
        """Oversample based on combinations of sensitive features."""
        # Create a composite key from all sensitive features
        available_features = [f for f in self.sensitive_features if f in df.columns]
        
        if not available_features:
            logger.warning("No sensitive features found in DataFrame")
            return df
        
        df = df.copy()
        df['_composite_key'] = df[available_features].apply(
            lambda row: '_'.join(str(v) for v in row), axis=1
        )
        
        result = self._oversample_by_column(df, '_composite_key', strategy)
        result = result.drop(columns=['_composite_key'])
        
        return result
    
    def smote_oversampling(self, df: pd.DataFrame, target_column: str,
                          feature_columns: List[str], k_neighbors: int = 5) -> pd.DataFrame:
        """
        Apply SMOTE (Synthetic Minority Oversampling Technique).
        
        Args:
            df: DataFrame to oversample
            target_column: Target column to balance
            feature_columns: List of feature columns to use for SMOTE
            k_neighbors: Number of neighbors for SMOTE algorithm
            
        Returns:
            Oversampled DataFrame with synthetic samples
        """
        try:
            from imblearn.over_sampling import SMOTE
            
            logger.info(f"Applying SMOTE oversampling with k_neighbors={k_neighbors}")
            
            X = df[feature_columns].values
            y = df[target_column].values
            
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Create new DataFrame
            result = pd.DataFrame(X_resampled, columns=feature_columns)
            result[target_column] = y_resampled
            
            # Add back other columns (using mode/mean for categorical/numerical)
            for col in df.columns:
                if col not in feature_columns and col != target_column:
                    if df[col].dtype == 'object':
                        result[col] = df[col].mode()[0]
                    else:
                        result[col] = df[col].mean()
            
            logger.info(f"Original size: {len(df)}, SMOTE size: {len(result)}")
            return result
            
        except ImportError:
            logger.error("imbalanced-learn not installed. Run: pip install imbalanced-learn")
            logger.info("Falling back to random oversampling")
            return self.random_oversampling(df, target_column, 'auto')
    
    def stratified_sampling(self, df: pd.DataFrame, sample_size: int,
                           stratify_columns: List[str] = None) -> pd.DataFrame:
        """
        Perform stratified sampling to maintain distribution.
        
        Args:
            df: DataFrame to sample
            sample_size: Number of samples to draw
            stratify_columns: Columns to stratify by (default: sensitive_features)
            
        Returns:
            Stratified sample DataFrame
        """
        if stratify_columns is None:
            stratify_columns = [f for f in self.sensitive_features if f in df.columns]
        
        if not stratify_columns:
            logger.warning("No stratification columns available, using random sampling")
            return df.sample(n=min(sample_size, len(df)), random_state=42)
        
        logger.info(f"Stratified sampling {sample_size} samples by {stratify_columns}")
        
        # Create composite stratification key
        df = df.copy()
        df['_strat_key'] = df[stratify_columns].apply(
            lambda row: '_'.join(str(v) for v in row), axis=1
        )
        
        # Calculate samples per stratum
        strata_counts = df['_strat_key'].value_counts()
        total = len(df)
        
        sampled_dfs = []
        for stratum, count in strata_counts.items():
            stratum_df = df[df['_strat_key'] == stratum]
            n_samples = int(sample_size * (count / total))
            n_samples = max(1, min(n_samples, len(stratum_df)))  # Ensure valid sample size
            
            sampled = stratum_df.sample(n=n_samples, random_state=42)
            sampled_dfs.append(sampled)
        
        result = pd.concat(sampled_dfs, ignore_index=True)
        result = result.drop(columns=['_strat_key'])
        
        logger.info(f"Stratified sample size: {len(result)}")
        return result
    
    def generate_fairness_report(self, original_df: pd.DataFrame,
                                mitigated_df: pd.DataFrame) -> Dict:
        """
        Generate a report comparing distributions before and after mitigation.
        
        Args:
            original_df: Original DataFrame
            mitigated_df: DataFrame after mitigation
            
        Returns:
            Dictionary containing comparison metrics
        """
        logger.info("Generating fairness report")
        
        report = {
            'original_size': len(original_df),
            'mitigated_size': len(mitigated_df),
            'features': {}
        }
        
        for feature in self.sensitive_features:
            if feature in original_df.columns and feature in mitigated_df.columns:
                orig_dist = original_df[feature].value_counts().to_dict()
                mit_dist = mitigated_df[feature].value_counts().to_dict()
                
                # Calculate imbalance ratios
                orig_values = list(orig_dist.values())
                mit_values = list(mit_dist.values())
                
                orig_ratio = max(orig_values) / min(orig_values) if min(orig_values) > 0 else float('inf')
                mit_ratio = max(mit_values) / min(mit_values) if min(mit_values) > 0 else float('inf')
                
                report['features'][feature] = {
                    'original_distribution': orig_dist,
                    'mitigated_distribution': mit_dist,
                    'original_imbalance_ratio': round(orig_ratio, 2),
                    'mitigated_imbalance_ratio': round(mit_ratio, 2),
                    'improvement': round(orig_ratio - mit_ratio, 2)
                }
                
                logger.info(f"{feature} - Imbalance ratio improved from {orig_ratio:.2f} to {mit_ratio:.2f}")
        
        return report


if __name__ == "__main__":
    try:
        print("=" * 80)
        print("BIAS MITIGATION WITH OVERSAMPLING TEST")
        print("=" * 80)
        
        # Create sample imbalanced dataset
        np.random.seed(42)
        
        data = {
            'gender': ['male'] * 70 + ['female'] * 20 + ['non-binary'] * 10,
            'age_group': np.random.choice(['young', 'adult', 'senior'], 100),
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        }
        
        df = pd.DataFrame(data)
        
        print("\nOriginal Dataset:")
        print("-" * 80)
        print(f"Total samples: {len(df)}")
        print(f"\nGender distribution:")
        print(df['gender'].value_counts())
        print(f"\nAge group distribution:")
        print(df['age_group'].value_counts())
        
        # Initialize bias mitigation
        mitigator = BiasMitigation(sensitive_features=['gender', 'age_group'])
        
        # Analyze original distribution
        print("\n" + "=" * 80)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        distribution = mitigator.analyze_distribution(df)
        for feature, stats in distribution.items():
            print(f"\n{feature.upper()}:")
            print("-" * 40)
            print(f"  Imbalance Ratio: {stats['imbalance_ratio']:.2f}")
            for value, pct in stats['percentages'].items():
                print(f"  {value}: {pct:.1f}%")
        
        # Apply random oversampling
        print("\n" + "=" * 80)
        print("RANDOM OVERSAMPLING")
        print("=" * 80)
        
        df_oversampled = mitigator.random_oversampling(df, strategy='auto')
        
        print(f"\nAfter Random Oversampling:")
        print("-" * 80)
        print(f"Total samples: {len(df_oversampled)}")
        print(f"\nGender distribution:")
        print(df_oversampled['gender'].value_counts())
        
        # Apply stratified sampling
        print("\n" + "=" * 80)
        print("STRATIFIED SAMPLING")
        print("=" * 80)
        
        df_stratified = mitigator.stratified_sampling(df_oversampled, sample_size=150)
        
        print(f"\nAfter Stratified Sampling:")
        print("-" * 80)
        print(f"Total samples: {len(df_stratified)}")
        print(f"\nGender distribution:")
        print(df_stratified['gender'].value_counts())
        
        # Generate fairness report
        print("\n" + "=" * 80)
        print("FAIRNESS REPORT")
        print("=" * 80)
        
        report = mitigator.generate_fairness_report(df, df_oversampled)
        
        print(f"\nOriginal size: {report['original_size']}")
        print(f"Mitigated size: {report['mitigated_size']}")
        
        for feature, metrics in report['features'].items():
            print(f"\n{feature.upper()}:")
            print("-" * 40)
            print(f"  Original imbalance ratio: {metrics['original_imbalance_ratio']}")
            print(f"  Mitigated imbalance ratio: {metrics['mitigated_imbalance_ratio']}")
            print(f"  Improvement: {metrics['improvement']}")
            print(f"\n  Original distribution:")
            for k, v in metrics['original_distribution'].items():
                print(f"    {k}: {v}")
            print(f"\n  Mitigated distribution:")
            for k, v in metrics['mitigated_distribution'].items():
                print(f"    {k}: {v}")
        
        print("\n" + "=" * 80)
        print("SMOTE OVERSAMPLING TEST")
        print("=" * 80)
        
        # Test SMOTE oversampling
        try:
            df_smote = mitigator.smote_oversampling(
                df, 
                target_column='target',
                feature_columns=['feature1', 'feature2'],
                k_neighbors=3
            )
            
            print(f"\nAfter SMOTE:")
            print("-" * 80)
            print(f"Total samples: {len(df_smote)}")
            print(f"\nTarget distribution:")
            print(df_smote['target'].value_counts())
        except Exception as e:
            print(f"\nSMOTE test skipped: {e}")
            print("Install imbalanced-learn to enable SMOTE: pip install imbalanced-learn")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
