from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.agents.core.agent import Agent, AgentInput, AgentOutput
from src.agents.eval_agent import EvaluationAgent
from src import logger


@dataclass
class BaselineEvaluationInput:
    df: pd.DataFrame
    target_column: str
    n_folds: int
    test_size: float
    model: str


@dataclass
class BaselineEvaluationOutput:
    original_eval: float
    original_nested_cv_scores: List[float]
    baseline_score: float


class BaselineEvaluationAgent(Agent):
    """Agent responsible for establishing baseline performance before augmentation."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def run(self, input: AgentInput) -> AgentOutput:
        # Handle input from previous agent (DataPreparationAgent)
        if hasattr(input.data, 'df'):
            # Input is a DataPreparationOutput object
            df = input.data.df
            # Get configuration from stored config or use defaults
            target_column = self.config.get("target_column", "class")
            n_folds = self.config.get("n_folds", 10)
            test_size = self.config.get("test_size", 0.2)
            model = self.config.get("model", "tabpfn")
        else:
            # Input is a dictionary (direct call)
            eval_input = BaselineEvaluationInput(**input.data)
            df = eval_input.df
            target_column = eval_input.target_column
            n_folds = eval_input.n_folds
            test_size = eval_input.test_size
            model = eval_input.model
        
        # Create evaluator
        evaluator = EvaluationAgent(
            label=target_column,
            n_folds=n_folds,
            test_size=test_size,
            model=model
        )
        
        # Test on holdout before any augmentation
        holdout_input = AgentInput(data={
            "df": eval_input.df,
            "target_column": eval_input.target_column,
            "evaluation_type": "kfold",
            "n_splits": eval_input.n_folds,
            "device": "cuda"
        })
        holdout_output = evaluator.run(holdout_input)
        original_eval = holdout_output.result.mean_score
        logger.debug(f"Original holdout evaluation: {original_eval}")
        
        # Nested cross-validation before any augmentation
        nested_cv_input = AgentInput(data={
            "df": eval_input.df,
            "target_column": eval_input.target_column,
            "evaluation_type": "nested_cv",
            "n_splits": eval_input.n_folds
        })
        nested_cv_output = evaluator.run(nested_cv_input)
        original_nested_cv_scores = nested_cv_output.result.scores
        baseline_score = nested_cv_output.result.mean_score
        logger.debug(f"Original nested CV scores: {baseline_score}")
        
        output = BaselineEvaluationOutput(
            original_eval=original_eval,
            original_nested_cv_scores=original_nested_cv_scores,
            baseline_score=baseline_score
        )
        
        return AgentOutput(
            result=output,
            metadata={"agent": "BaselineEvaluationAgent", "status": "success"}
        )
