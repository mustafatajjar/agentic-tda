from autogluon.core.data import LabelCleaner
from autogluon.core.models import BaggedEnsembleModel
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularPredictor
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from skrub import TableVectorizer
from tabicl import TabICLClassifier
from sklearn.pipeline import make_pipeline

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
        self.kf = KFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        self.fold_indices = [
            test_idx for _, test_idx in self.kf.split(data.iloc[self.train_indices])
        ]

    def _split_train_test_indices(self, data):
        train_idx, test_idx = train_test_split(
            np.arange(len(data)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=data[self.label] if self.label in data else None,
        )
        return train_idx, test_idx

    def nested_cross_val(self, data, method="tabpfn"):
        """Performs nested cross-validation on the training data using the specified method."""
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

            if method == "lightgbm":
                model = LGBMClassifier(
                    verbose=-1,
                    random_state=self.random_state,
                )
                model.fit(X_train, y_train)
                y_score = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_score)
                scores.append(score)

            elif method == "tabpfn":
                clf = TabPFNClassifier(device="cuda")
                clf.fit(X_train.values, y_train)
                y_score = clf.predict_proba(X_val.values)[:, 1]
                score = roc_auc_score(y_val, y_score)
                scores.append(score)

            else:
                raise ValueError(f"Unknown method: {method}")

        return scores

    def test_on_holdout(self, data, time_limit=60, method="tabpfn"):
        """
        Test on the holdout test set (from initial split) using the provided DataFrame.
        method: "lightgbm" or "tabpfn"
        """
        train_data = data.iloc[self.train_indices].copy()
        test_data = data.iloc[self.test_indices].copy()

        numeric_cols = train_data.select_dtypes(exclude="category").columns
        train_data[numeric_cols] = train_data[numeric_cols].fillna(0)
        test_data[numeric_cols] = test_data[numeric_cols].fillna(0)

        if method == "lightgbm":
            X_train = train_data.drop(columns=[self.label])
            y_train = train_data[self.label]
            X_test = test_data.drop(columns=[self.label])
            y_test = test_data[self.label]

            # Convert object columns to category
            for col in X_train.columns:
                if X_train[col].dtype == "object":
                    X_train[col] = X_train[col].astype("category")
                    X_test[col] = X_test[col].astype("category")

            # Convert label to numeric if needed
            if y_train.dtype == "object":
                y_train = pd.Categorical(y_train).codes
            if y_test.dtype == "object":
                y_test = pd.Categorical(y_test).codes

            model = LGBMClassifier(
                verbose=-1,
                random_state=self.random_state,
            )

            model.fit(X_train, y_train)
            y_score = model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_score)
            return score

        elif method == "tabpfn":
            X_train = train_data.drop(columns=[self.label])
            y_train = train_data[self.label]
            X_test = test_data.drop(columns=[self.label])
            y_test = test_data[self.label]

            # Convert categorical columns to category dtype
            for col in X_train.columns:
                if X_train[col].dtype == "object":
                    X_train[col] = X_train[col].astype("category")
                    X_test[col] = X_test[col].astype("category")

            # Convert label to numeric if needed
            if y_train.dtype == "object":
                y_train = pd.Categorical(y_train).codes
            if y_test.dtype == "object":
                y_test = pd.Categorical(y_test).codes

            clf = TabPFNClassifier(device="cuda")
            clf.fit(X_train.values, y_train)
            y_score = clf.predict_proba(X_test.values)[:, 1]
            score = roc_auc_score(y_test, y_score)
            return score

        else:
            raise ValueError(f"Unknown method: {method}")

    def test_on_holdout_kfold_tabpfn(self, data=None, n_splits=5, device="cpu", method="tabpfn"):
        """
        Perform KFold cross-validation on all the data using TabPFN or LightGBM.
        Returns a list of ROC AUC scores for each fold.
        method: "tabpfn" or "lightgbm"
        """
        if data is None:
            data = self.data
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        X = data.drop(columns=[self.label])
        y = data[self.label]
        scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Convert object columns to category codes
            for col in X_train.columns:
                if X_train[col].dtype == "object":
                    X_train[col] = X_train[col].astype("category")
                    X_test[col] = X_test[col].astype("category")

            # Convert label to numeric if needed
            if y_train.dtype == "object":
                y_train = pd.Categorical(y_train).codes
            if y_test.dtype == "object":
                y_test = pd.Categorical(y_test).codes

            if method == "tabpfn":
                clf = TabPFNClassifier(device=device)
                clf.fit(X_train.values, y_train)
                y_score = clf.predict_proba(X_test.values)[:, 1]
                score = roc_auc_score(y_test, y_score)
                scores.append(score)
            elif method == "lightgbm":
                model = LGBMClassifier(
                    verbose=-1,
                    random_state=self.random_state,
                )
                model.fit(X_train, y_train)
                y_score = model.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_score)
                scores.append(score)
            else:
                raise ValueError(f"Unknown method: {method}")

        return scores

    def get_fold_indices(self):
        """Return the indices for each fold (relative to the train set)."""
        return self.fold_indices

    def get_train_test_indices(self):
        """Return the train and test indices from the initial split."""
        return self.train_indices, self.test_indices

class TabPFNSubsampleWrapper:
    """
    A wrapper for TabPFNClassifier that subsamples the training data to 10,000 rows if more are provided.
    """
    def __init__(self, device="cpu", **kwargs):
        self.device = device
        self.kwargs = kwargs
        self.model = TabPFNClassifier(device=device, **kwargs)

    def fit(self, X, y):
        # Subsample if more than 10,000 rows
        if X.shape[0] > 10000:
            idx = np.random.choice(X.shape[0], 10000, replace=False)
            X_sub = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx]
            y_sub = y[idx] if isinstance(y, np.ndarray) else y.iloc[idx]
        else:
            X_sub = X
            y_sub = y
        self.model.fit(X_sub, y_sub)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)