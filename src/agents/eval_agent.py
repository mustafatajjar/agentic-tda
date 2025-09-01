from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from src.agents.core.agent import Agent, AgentInput, AgentOutput
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from src import logger

pd.set_option("mode.use_inf_as_na", True)


@dataclass
class EvaluationInput:
    df: pd.DataFrame
    target_column: str
    evaluation_type: str = "nested_cv"  # "nested_cv", "holdout", "kfold"
    n_splits: int = 5
    device: str = "cpu"


@dataclass
class EvaluationOutput:
    scores: List[float]
    mean_score: float
    std_score: float
    evaluation_type: str
    n_splits: int


class EvaluationAgent(Agent):
    """
    Unified evaluation agent that handles different types of model evaluation.
    Supports nested cross-validation, holdout testing, and k-fold cross-validation.
    """
    
    def __init__(
        self,
        label: str = "class",
        test_size: float = 0.2,
        n_folds: int = 5,
        random_state: int = 42,
        model: str = "tabpfn",
    ):
        self.label = label
        self.test_size = test_size
        self.n_folds = n_folds
        self.random_state = random_state
        self.model_type = model
        
        # Initialize model based on type
        self._initialize_model()
        
        # Store indices for later use
        self.train_indices = None
        self.test_indices = None
        self.fold_indices = None
    
    def _initialize_model(self):
        """Initialize the ML model based on the specified type."""
        if self.model_type == "lightgbm":
            self.model = LGBMClassifier(
                verbose=-1,
                random_state=self.random_state,
            )
            self.use_values = False  # Use DataFrame directly
        elif self.model_type == "tabpfn":
            self.model = TabPFNClassifier(device="cpu")
            self.use_values = True  # Use .values for TabPFN
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Supported: 'lightgbm', 'tabpfn'")
    
    def run(self, input: AgentInput) -> AgentOutput:
        """Main entry point for the evaluation agent."""
        eval_input = EvaluationInput(**input.data)
        
        # Set up indices if not already done
        if self.train_indices is None:
            self._setup_indices(eval_input.df)
        
        # Perform evaluation based on type
        if eval_input.evaluation_type == "nested_cv":
            scores = self._nested_cross_validation(eval_input.df)
        elif eval_input.evaluation_type == "holdout":
            scores = [self._holdout_evaluation(eval_input.df)]
        elif eval_input.evaluation_type == "kfold":
            scores = self._kfold_cross_validation(eval_input.df, eval_input.n_splits)
        else:
            raise ValueError(f"Unknown evaluation type: {eval_input.evaluation_type}")
        
        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        output = EvaluationOutput(
            scores=scores,
            mean_score=mean_score,
            std_score=std_score,
            evaluation_type=eval_input.evaluation_type,
            n_splits=len(scores)
        )
        
        logger.debug(f"Evaluation completed: {eval_input.evaluation_type}, "
                    f"mean_score={mean_score:.4f}, std_score={std_score:.4f}")
        
        return AgentOutput(
            result=output,
            metadata={
                "agent": "EvaluationAgent",
                "model_type": self.model_type,
                "status": "success"
            }
        )
    
    def _setup_indices(self, data: pd.DataFrame):
        """Set up train/test and fold indices."""
        # Initial train/test split
        self.train_indices, self.test_indices = self._split_train_test_indices(data)
        
        # K-fold split for nested CV
        kf = KFold(
            n_splits=self.n_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        self.fold_indices = [
            test_idx for _, test_idx in kf.split(data.iloc[self.train_indices])
        ]
    
    def _split_train_test_indices(self, data: pd.DataFrame):
        """Split data into train and test indices."""
        train_idx, test_idx = train_test_split(
            np.arange(len(data)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=data[self.label] if self.label in data else None,
        )
        return train_idx, test_idx
    
    def _nested_cross_validation(self, data: pd.DataFrame) -> List[float]:
        """Perform nested cross-validation on the training data."""
        if self.train_indices is None or self.fold_indices is None:
            raise ValueError("Indices not set up. Call _setup_indices first.")
        
        scores = []
        train_data = data.iloc[self.train_indices].reset_index(drop=True)
        
        for fold, test_idx in enumerate(self.fold_indices):
            val_data = train_data.iloc[test_idx].copy()
            train_fold = train_data.drop(index=test_idx).copy()
            
            score = self._evaluate_fold(train_fold, val_data)
            scores.append(score)
            
            logger.debug(f"Fold {fold + 1}/{self.n_folds}: score={score:.4f}")
        
        return scores
    
    def _holdout_evaluation(self, data: pd.DataFrame) -> float:
        """Evaluate on the holdout test set."""
        if self.train_indices is None or self.test_indices is None:
            raise ValueError("Indices not set up. Call _setup_indices first.")
        
        train_data = data.iloc[self.train_indices].copy()
        test_data = data.iloc[self.test_indices].copy()
        
        score = self._evaluate_fold(train_data, test_data)
        logger.debug(f"Holdout evaluation: score={score:.4f}")
        
        return score
    
    def _kfold_cross_validation(self, data: pd.DataFrame, n_splits: int) -> List[float]:
        """Perform k-fold cross-validation on all data."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scores = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            train_fold = data.iloc[train_idx].copy()
            test_fold = data.iloc[test_idx].copy()
            
            score = self._evaluate_fold(train_fold, test_fold)
            scores.append(score)
            
            logger.debug(f"KFold {fold + 1}/{n_splits}: score={score:.4f}")
        
        return scores
    
    def _evaluate_fold(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> float:
        """Evaluate a single fold by training on train_data and testing on test_data."""
        # Prepare features and target
        X_train = train_data.drop(columns=[self.label])
        y_train = train_data[self.label]
        X_test = test_data.drop(columns=[self.label])
        y_test = test_data[self.label]
        
        # Preprocess data
        X_train, X_test = self._preprocess_features(X_train, X_test)
        y_train, y_test = self._preprocess_target(y_train, y_test)
        
        # Train and predict
        if self.use_values:
            self.model.fit(X_train.values, y_train)
            y_score = self.model.predict_proba(X_test.values)[:, 1]
        else:
            self.model.fit(X_train, y_train)
            y_score = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate score
        score = roc_auc_score(y_test, y_score)
        return score
    
    def _preprocess_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Preprocess features for training and testing."""
        # Fill NA for numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train[numeric_cols] = X_train[numeric_cols].fillna(0)
        X_test[numeric_cols] = X_test[numeric_cols].fillna(0)
        
        # Convert object columns to category
        for col in X_train.columns:
            if X_train[col].dtype == "object":
                X_train[col] = X_train[col].astype("category")
                X_test[col] = X_test[col].astype("category")
        
        return X_train, X_test
    
    def _preprocess_target(self, y_train: pd.Series, y_test: pd.Series):
        """Preprocess target variables."""
        # Convert to numeric if needed
        if y_train.dtype == "object":
            y_train = pd.Categorical(y_train).codes
        if y_test.dtype == "object":
            y_test = pd.Categorical(y_test).codes
        
        return y_train, y_test
    
    # Convenience methods for backward compatibility
    def nested_cross_val(self, data: pd.DataFrame) -> List[float]:
        """Backward compatibility method for nested cross-validation."""
        if self.train_indices is None:
            self._setup_indices(data)
        return self._nested_cross_validation(data)
    
    def test_on_holdout_kfold(self, data: pd.DataFrame, n_splits: int = 5, device: str = "cpu") -> List[float]:
        """Backward compatibility method for k-fold cross-validation."""
        return self._kfold_cross_validation(data, n_splits)
    
    def test_on_holdout(self, data: pd.DataFrame, time_limit: int = 60) -> float:
        """Backward compatibility method for holdout evaluation."""
        if self.train_indices is None:
            self._setup_indices(data)
        return self._holdout_evaluation(data)
    
    def get_fold_indices(self) -> List[np.ndarray]:
        """Return the indices for each fold (relative to the train set)."""
        if self.fold_indices is None:
            raise ValueError("Indices not set up. Call _setup_indices first.")
        return self.fold_indices
    
    def get_train_test_indices(self) -> tuple:
        """Return the train and test indices from the initial split."""
        if self.train_indices is None or self.test_indices is None:
            raise ValueError("Indices not set up. Call _setup_indices first.")
        return self.train_indices, self.test_indices
