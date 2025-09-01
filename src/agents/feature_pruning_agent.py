from dataclasses import dataclass
from typing import Dict, Any, List
from collections import Counter
from itertools import chain
import numpy as np
import pandas as pd

from src.agents.core.agent import Agent, AgentInput, AgentOutput
from src.utils.funcs import prune_features_binary_classification
from src import logger


@dataclass
class FeaturePruningInput:
    df: pd.DataFrame
    target_column: str


@dataclass
class FeaturePruningOutput:
    pruned_df: pd.DataFrame
    selected_features: List[str]
    pruning_effective: bool
    original_feature_count: int
    pruned_feature_count: int


class FeaturePruningAgent(Agent):
    """Agent responsible for feature selection and pruning after augmentation."""
    
    def run(self, input: AgentInput) -> AgentOutput:
        pruning_input = FeaturePruningInput(**input.data)
        
        try:
            y = pruning_input.df[pruning_input.target_column]
            X = pruning_input.df.drop(columns=[pruning_input.target_column])
            
            original_feature_count = len(X.columns)
            
            # Perform feature pruning
            selected_features_per_split = prune_features_binary_classification(X, y)
            
            # Combine features selected across splits (majority vote)
            feature_counts = Counter(
                chain.from_iterable(selected_features_per_split)
            )
            num_splits = len(selected_features_per_split)
            selected_features = [
                f
                for f, count in feature_counts.items()
                if count >= (num_splits // 2 + 1)
            ]
            
            logger.debug(f"Selected features: {selected_features}")
            logger.debug(f"X columns: {X.columns.tolist()}")
            
            # Validate selected features
            if isinstance(selected_features, str):
                selected_features = [selected_features]
            
            if not all(f in X.columns for f in selected_features):
                logger.debug("WARNING: Some selected features are not in DataFrame columns!")
                # Fall back to original features
                selected_features = X.columns.tolist()
            
            # Apply pruning if effective
            pruning_effective = False
            if len(selected_features) < len(X.columns) and all(
                f in X.columns for f in selected_features
            ):
                logger.debug(
                    f"Pruning effective: {len(X.columns)} -> {len(selected_features)} features."
                )
                X_pruned = X[selected_features]
                pruned_df = X_pruned.copy()
                pruned_df[pruning_input.target_column] = y
                pruning_effective = True
            else:
                logger.debug(
                    "Pruning did not remove any features, using original augmented dataframe."
                )
                pruned_df = pruning_input.df
                selected_features = X.columns.tolist()
            
            pruned_feature_count = len(selected_features)
            
            output = FeaturePruningOutput(
                pruned_df=pruned_df,
                selected_features=selected_features,
                pruning_effective=pruning_effective,
                original_feature_count=original_feature_count,
                pruned_feature_count=pruned_feature_count
            )
            
            return AgentOutput(
                result=output,
                metadata={"agent": "FeaturePruningAgent", "status": "success"}
            )
            
        except Exception as e:
            logger.debug(f"Feature pruning failed: {e}")
            # Return original dataframe if pruning fails
            output = FeaturePruningOutput(
                pruned_df=pruning_input.df,
                selected_features=[],
                pruning_effective=False,
                original_feature_count=len(pruning_input.df.columns) - 1,
                pruned_feature_count=len(pruning_input.df.columns) - 1
            )
            
            return AgentOutput(
                result=output,
                metadata={"agent": "FeaturePruningAgent", "status": "failed", "error": str(e)}
            )
