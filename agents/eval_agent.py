import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_arff_to_dataframe(file_path):
    """
    Load ARFF file into pandas DataFrame
    
    Parameters:
    - file_path: path to the ARFF file
    
    Returns:
    - pandas DataFrame
    """
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    # Decode byte strings to regular strings if needed
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    
    for col in str_df:
        df[col] = str_df[col]
        
    return df

def evaluate(table):
    # preprocess data
    df = load_arff_to_dataframe("data/")
    X = df.drop(columns=["binaryClass"])
    y = df["binaryClass"]
    

    # fit model
    model = RandomForestClassifier()
    
    # predict

    # calculate score
    pass


''' Implementation Suggestion:

class EvaluationAgent:
    def __init__(self):
        self.metrics_history = []
    
    def compare_tables(self, original_df, augmented_df, target_column, 
                     test_size=0.2, random_state=42):
        """
        Compare model performance between original and augmented DataFrames
        
        Args:
            original_df: DataFrame before augmentation
            augmented_df: DataFrame after augmentation
            target_column: Name of the target variable
            test_size: Proportion for test split (default: 0.2)
            random_state: Random seed (default: 42)
            
        Returns:
            Dictionary with comparison results
        """
        # Validate input shapes
        assert len(original_df) == len(augmented_df), "DataFrames must have same row count"
        
        # 1. Evaluate original data
        X_orig = original_df.drop(columns=[target_column])
        y_orig = original_df[target_column]
        orig_metrics = self._compute_metrics(X_orig, y_orig, test_size, random_state)
        
        # 2. Evaluate augmented data
        X_aug = augmented_df.drop(columns=[target_column])
        y_aug = augmented_df[target_column]
        aug_metrics = self._compute_metrics(X_aug, y_aug, test_size, random_state)
        
        # 3. Compare feature importance
        fi_comparison = self._compare_feature_importance(
            orig_metrics['feature_importance'],
            aug_metrics['feature_importance']
        )
        
        # Store results
        result = {
            'baseline': orig_metrics,
            'augmented': aug_metrics,
            'improvement': {
                'accuracy': aug_metrics['accuracy'] - orig_metrics['accuracy'],
                'f1': aug_metrics['f1'] - orig_metrics['f1']
            },
            'feature_impact': fi_comparison
        }
        self.metrics_history.append(result)
        return result

    def _compute_metrics(self, X, y, test_size, random_state):
        """Internal metric computation"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }

    def _compare_feature_importance(self, orig_fi, aug_fi):
        """Compare feature importance between models"""
        all_features = set(orig_fi.keys()).union(aug_fi.keys())
        return pd.DataFrame([
            {
                'feature': feat,
                'original_importance': orig_fi.get(feat, 0),
                'augmented_importance': aug_fi.get(feat, 0),
                'delta': aug_fi.get(feat, 0) - orig_fi.get(feat, 0)
            }
            for feat in all_features
        ]).sort_values('delta', ascending=False)
'''