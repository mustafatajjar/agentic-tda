from dataclasses import dataclass
from typing import Dict, Any, Tuple
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.agents.core.agent import Agent, AgentInput, AgentOutput
from src.utils.funcs import arff_to_dataframe, extract_arff_metadata
from src import logger


@dataclass
class DataPreparationInput:
    data_path: str
    n_folds: int = 10
    random_state: int = 42


@dataclass
class DataPreparationOutput:
    df: pd.DataFrame
    metadata: str
    fold_indices: list
    original_columns: list


class DataPreparationAgent(Agent):
    """Agent responsible for loading data, extracting metadata, and preparing cross-validation folds."""
    
    def run(self, input: AgentInput) -> AgentOutput:
        prep_input = DataPreparationInput(**input.data)
        
        logger.debug(f"Loading ARFF file from: {prep_input.data_path}")
        
        # Load data and metadata
        df = arff_to_dataframe(prep_input.data_path)
        metadata = extract_arff_metadata(prep_input.data_path)
        
        # Create cross-validation folds
        kf = KFold(n_splits=prep_input.n_folds, shuffle=True, random_state=prep_input.random_state)
        fold_indices = [list(test_index) for _, test_index in kf.split(df)]
        
        # Save fold indices for reproducibility
        os.makedirs("outputs", exist_ok=True)
        for i, indices in enumerate(fold_indices):
            with open(f"outputs/fold_{i+1}_indices.txt", "w") as f:
                f.write("\n".join(map(str, indices)))
        
        output = DataPreparationOutput(
            df=df,
            metadata=metadata,
            fold_indices=fold_indices,
            original_columns=df.columns.tolist()
        )
        
        return AgentOutput(
            result=output,
            metadata={"agent": "DataPreparationAgent", "status": "success"}
        )
