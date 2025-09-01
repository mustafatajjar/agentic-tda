from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

pd.set_option("mode.use_inf_as_na", True)


class EvaluationAgent:
    def __init__(
        self,
        data,
        label="class",
        test_size=0.2,
        n_folds=5,
        random_state=42,
        model="tabpfn",
    ):
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

        # Instantiate the model once in the constructor
        if model == "lightgbm":
            self.model = LGBMClassifier(
                verbose=-1,
                random_state=self.random_state,
            )
            self.use_values = False  # Use DataFrame directly
        elif model == "tabpfn":
            self.model = TabPFNClassifier(device="cpu")
            self.use_values = True  # Use .values for TabPFN
        else:
            raise ValueError(f"Unknown model: {model}")

    def _split_train_test_indices(self, data):
        train_idx, test_idx = train_test_split(
            np.arange(len(data)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=data[self.label] if self.label in data else None,
        )
        return train_idx, test_idx

    def nested_cross_val(self, data):
        """Performs nested cross-validation on the training data using the specified model."""
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

            # Fit and predict using the instantiated model
            if self.use_values:
                self.model.fit(X_train.values, y_train)
                y_score = self.model.predict_proba(X_val.values)[:, 1]
            else:
                self.model.fit(X_train, y_train)
                y_score = self.model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_score)
            scores.append(score)

        return scores

    def test_on_holdout(self, data, time_limit=60):
        """
        Test on the holdout test set (from initial split) using the provided DataFrame.
        Uses the model specified in the constructor.
        """
        train_data = data.iloc[self.train_indices].copy()
        test_data = data.iloc[self.test_indices].copy()

        numeric_cols = train_data.select_dtypes(exclude="category").columns
        train_data[numeric_cols] = train_data[numeric_cols].fillna(0)
        test_data[numeric_cols] = test_data[numeric_cols].fillna(0)

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

        # Fit and predict using the instantiated model
        if self.use_values:
            self.model.fit(X_train.values, y_train)
            y_score = self.model.predict_proba(X_test.values)[:, 1]
        else:
            self.model.fit(X_train, y_train)
            y_score = self.model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_score)
        return score

    def test_on_holdout_kfold(self, data=None, n_splits=5, device="cpu"):
        """
        Perform KFold cross-validation on all the data using TabPFN or LightGBM.
        Returns a list of ROC AUC scores for each fold.
        Uses the model specified in the constructor.
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

            # Fit and predict using the instantiated model
            if self.use_values:
                self.model.fit(X_train.values, y_train)
                y_score = self.model.predict_proba(X_test.values)[:, 1]
            else:
                self.model.fit(X_train, y_train)
                y_score = self.model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_score)
            scores.append(score)

        return scores

    def get_fold_indices(self):
        """Return the indices for each fold (relative to the train set)."""
        return self.fold_indices

    def get_train_test_indices(self):
        """Return the train and test indices from the initial split."""
        return self.train_indices, self.test_indices
