import time
import warnings
import pandas as pd
from lightgbm import LGBMClassifier
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.core.utils.feature_selection import FeatureSelector
from autogluon.core.models import AbstractModel
from autogluon.common.utils.log_utils import set_logger_verbosity

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
set_logger_verbosity(4)
    
def prune_features_binary_classification(
    X: "pd.DataFrame", 
    y: "pd.Series", 
    *, 
    time_limit_per_split: int = 600, 
    eval_metric="accuracy"
) -> list:
    """
    Prunes uninformative features using AutoGluon's FeatureSelector with LightGBM.

    Args:
        X: Input features (numeric or categorical).
        y: Target labels (binary classification).
        time_limit_per_split: Max time in seconds allowed for pruning.
        eval_metric: Metric to optimize during pruning.

    Returns:
        List of pruned (retained) feature names.
    """
    # AutoGluon AbstractModel wrapper for LGBM
    class FEModel(AbstractModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._feature_generator = None

        def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
            from lightgbm import LGBMClassifier
            from autogluon.features.generators import AutoMLPipelineFeatureGenerator

            X = self.preprocess(X, is_train=True)

            # Encode categoricals
            self._feature_generator = AutoMLPipelineFeatureGenerator(verbosity=0)
            X = self._feature_generator.fit_transform(X, y)

            self.model = LGBMClassifier(random_state=42, verbose=-1)
            self.model.fit(X, y)

        def predict(self, X, **kwargs):
            X = self.preprocess(X)
            if self._feature_generator:
                X = self._feature_generator.transform(X)
            return self.model.predict(X)

        def _set_default_params(self):
            self._set_default_param_value("random_state", 42)

    print(f"Pruning from {X.shape[1]} features...")

    # FeatureSelector configuration
    selection_config = dict(
        n_fi_subsample=100000,
        prune_threshold=0.0,  # Removes features with no positive importance
        prune_ratio=0.1,  # stricter pruning
        stopping_round=20,
    )

    fs = FeatureSelector(
        model=FEModel(eval_metric=eval_metric),
        time_limit=time_limit_per_split,
        problem_type="binary",
        seed=0,
        raise_exception=True,
    )

    candidate_features = fs.select_features(
        X=X,
        y=y,
        X_val=X.copy(),
        y_val=y.copy(),
        n_train_subsample=50000,
        min_improvement=0,
        **selection_config,
    )

    print(f"Final retained features ({len(candidate_features)}): {candidate_features}")
    removed_features = [f for f in X.columns if f not in candidate_features]
    print(f"Features removed ({len(removed_features)}): {removed_features}")
    return candidate_features