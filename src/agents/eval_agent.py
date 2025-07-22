from autogluon.core.data import LabelCleaner
from autogluon.core.models import BaggedEnsembleModel
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularPredictor
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd

pd.set_option("mode.use_inf_as_na", True)


class EvaluationAgent:
    def __init__(self, data, label="class", test_size=0.2, n_folds=5, random_state=42):
        self.data = data
        self.label = label
        self.test_size = test_size
        self.n_folds = n_folds
        self.random_state = random_state

        # Split into train/test once, store indices
        self.train_indices, self.test_indices = self._split_train_test_indices(data)
        self.kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.fold_indices = [test_idx for _, test_idx in self.kf.split(data.iloc[self.train_indices])]

    def _split_train_test_indices(self, data):
        train_idx, test_idx = train_test_split(
            np.arange(len(data)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=data[self.label] if self.label in data else None
        )
        return train_idx, test_idx

    def nested_cross_val(self, data):
        """Performs nested cross-validation on the training data using LightGBM with lgb.Dataset."""
        scores = []
        train_data = data.iloc[self.train_indices].reset_index(drop=True)
        X = train_data.drop(columns=[self.label])
        y = train_data[self.label]

        for fold, test_idx in enumerate(self.fold_indices):
            val_data = train_data.iloc[test_idx].copy()
            train_fold = train_data.drop(index=test_idx).copy()

            X_train = train_fold.drop(columns=[self.label])
            y_train = train_fold[self.label]
            X_val = val_data.drop(columns=[self.label])
            y_val = val_data[self.label]

            # Fill NA for numeric columns
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            X_train[numeric_cols] = X_train[numeric_cols].fillna(0)
            X_val[numeric_cols] = X_val[numeric_cols].fillna(0)

            # Convert object columns to category
            for col in X_train.columns:
                if X_train[col].dtype == "object":
                    X_train[col] = X_train[col].astype("category")
                    X_val[col] = X_val[col].astype("category")

            # Convert label to numeric if needed
            if y_train.dtype == "object":
                y_train = pd.Categorical(y_train).codes
            if y_val.dtype == "object":
                y_val = pd.Categorical(y_val).codes

            # Create lgb.Dataset objects
            lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=False)

            params = {
                "boosting_type": "gbdt",
                "objective": "binary",
                "metric": "auc",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "random_state": self.random_state,
            }

            gbm = lgb.train(
                params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_val],
            )

            y_score = gbm.predict(X_val)
            score = roc_auc_score(y_val, y_score)
            scores.append(score)
        return scores

    def test_on_holdout(self, data):
        """Test on the holdout test set (from initial split) using the provided DataFrame."""
        train_data = data.iloc[self.train_indices].copy()
        test_data = data.iloc[self.test_indices].copy()

        numeric_cols = train_data.select_dtypes(exclude="category").columns
        train_data[numeric_cols] = train_data[numeric_cols].fillna(0)
        test_data[numeric_cols] = test_data[numeric_cols].fillna(0)

        predictor = TabularPredictor(
            label=self.label, eval_metric="roc_auc"
        ).fit(
            train_data=train_data,presets="good_quality", verbosity=0,time_limit=60*1,
        )

        y_true = test_data[self.label].to_numpy()
        y_score = predictor.predict_proba(test_data)[predictor.positive_class].to_numpy()
        score = roc_auc_score(y_true, y_score)
        return score

    def get_fold_indices(self):
        """Return the indices for each fold (relative to the train set)."""
        return self.fold_indices

    def get_train_test_indices(self):
        """Return the train and test indices from the initial split."""
        return self.train_indices, self.test_indices
