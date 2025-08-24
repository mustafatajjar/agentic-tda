import time
import warnings
import json
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.core.utils.feature_selection import FeatureSelector, logger
from autogluon.core.models import AbstractModel
from autogluon.common.utils.log_utils import set_logger_verbosity
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from tabpfn import TabPFNClassifier


set_logger_verbosity(verbosity=1)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def prune_features_binary_classification(
    X: "pd.DataFrame",
    y: "pd.DataFrame",
    *,
    time_limit_per_split: int = 3600,
    cv="default",
    eval_metric="accuracy",
):
    """Obtain the optimal set of features for a given dataset by iterative (clever)
    feature pruning with AutoGluon and LightGBM.

    This will try to find the optimal set of features to improve predictive performance
    using the CV strategy provided.

    Requirements:
        - autogluon

    Args:
        X: The input features.
        y: The target variable.
        time_limit_per_split: The time limit in seconds.
            This much time is used at most per split of the CV.
        cv: The splitter to use for the cross-validation.
            If "default", RepeatedStratifiedKFold with 3 splits and 100 repeats is used.
        eval_metric: The evaluation metric to use for the model.
            Change this to a supported sklearn/autogluon metric to optimize w.r.t.
            your metric of interest.
    """

    class FEModel(AbstractModel):
        def __init__(self, **kwargs):
            kwargs.setdefault("path", "models/temp")
            super().__init__(**kwargs)
            self._feature_generator = None

        def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
            X = self.preprocess(X, is_train=True)
            self._feature_generator = AutoMLPipelineFeatureGenerator(verbosity=0)
            X = self._feature_generator.fit_transform(X, y)
            self.model = LGBMClassifier(random_state=42, verbose=-1)
            self.model.fit(X, y)

        def predict(self, X, **kwargs) -> np.ndarray:
            X = self.preprocess(X)
            if self._feature_generator:
                X = self._feature_generator.transform(X)
            return self.model.predict(X)

        def _set_default_params(self):
            self._set_default_param_value("random_state", 42)

    if cv == "default":
        splitter = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=42)
    else:
        splitter = cv

    time_limit = time_limit_per_split
    optimal_features_per_split = []

    for train_index, test_index in splitter.split(X=X, y=y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        feature_generator = AutoMLPipelineFeatureGenerator(verbosity=0)
        X_train_transformed = feature_generator.fit_transform(X=X_train, y=y_train)
        X_train = X_train[X_train_transformed.columns]

        logger.info(
            f"AutoGluon Useless Feature Pruning:\tPruned from {X_train.shape[1]} to {X_train_transformed.shape[1]} features."
        )

        st_time = time.time()
        candidate_features = X_train.columns.tolist()

        for selection_step, selection_config in enumerate(
            [
                dict(
                    n_fi_subsample=10000,
                    prune_threshold="noise",
                    prune_ratio=0.075,
                    stopping_round=3,
                ),
                dict(
                    n_fi_subsample=100000,
                    prune_threshold="none",
                    prune_ratio=0.05,
                    stopping_round=20,
                ),
                dict(
                    n_fi_subsample=500000,
                    prune_threshold="none",
                    prune_ratio=0.025,
                    stopping_round=40,
                ),
            ]
        ):

            rest_time = time_limit - (time.time() - st_time)
            logger.info(
                f"AutoGluon Feature Pruning {selection_step} | Time Left: {rest_time:.2f} seconds."
            )

            fs = FeatureSelector(
                model=FEModel(eval_metric=eval_metric),
                time_limit=time_limit_per_split,
                problem_type="binary",
                seed=0,
                raise_exception=True,
            )

            candidate_features = fs.select_features(
                X=X_train,
                y=y_train,
                X_val=X_test,
                y_val=y_test,
                n_train_subsample=50000,
                min_improvement=0,
                **selection_config,
            )

            X_train = X_train[candidate_features]
            X_test = X_test[candidate_features]

        logger.info(
            f"AutoGluon Feature Pruning:\tPruned to {X_train.shape[1]} features."
        )
        logger.info(f"Final Features: {candidate_features}")
        optimal_features_per_split.append(candidate_features)

        with open("optimal_features_per_split_im.json", "w") as file:
            json.dump(optimal_features_per_split, file)

    logger.info(f"Optimal Features Per Split: {optimal_features_per_split}")
    with open("optimal_features_per_split.json", "w") as file:
        json.dump(optimal_features_per_split, file)

    return optimal_features_per_split


def prune_features_sfs(
    X: pd.DataFrame,
    y: pd.Series,
    direction: str = "forward",
    scoring: str = "roc_auc",
    cv=3,
    device: str = "cuda",
    method: str = "tabpfn",  # "tabpfn" or "lightgbm"
):
    """
    Prune features using sklearn's SequentialFeatureSelector with TabPFN or LightGBM as the estimator.

    Args:
        X: Input features (DataFrame).
        y: Target variable (Series).
        n_features_to_select: Number of features to select (default: half of features).
        direction: "forward" or "backward" selection.
        scoring: Scoring metric for selection.
        cv: Number of cross-validation folds.
        device: "cpu" or "cuda" for TabPFN.
        method: "tabpfn" or "lightgbm"

    Returns:
        selected_features: List of selected feature names.
    """
    # Ensure all non-numeric columns are encoded as categorical codes
    X_enc = X.copy()
    for col in X_enc.columns:
        if not pd.api.types.is_numeric_dtype(X_enc[col]):
            X_enc[col] = pd.Categorical(X_enc[col]).codes
    X_enc = X_enc.astype(np.float64)

    if method == "tabpfn":
        estimator = TabPFNClassifier(device="cuda")
    elif method == "lightgbm":
        estimator = LGBMClassifier(
            verbose=-1,
            random_state=42,
        )
    else:
        raise ValueError("method must be 'tabpfn' or 'lightgbm'")

    sfs = SequentialFeatureSelector(
        estimator,
        scoring=scoring,
        cv=cv,
        n_jobs=8,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        sfs.fit(X_enc, y)
    selected_features = X_enc.columns[sfs.get_support()].tolist()

    return selected_features
