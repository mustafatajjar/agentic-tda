from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.agents.core.agent import Agent, AgentInput, AgentOutput
from src.agents.eval_agent import EvaluationAgent
from src import logger


@dataclass
class PerformanceTrackingInput:
    df: pd.DataFrame
    target_column: str
    n_folds: int
    test_size: float
    model: str
    current_iteration: int
    max_iterations: int
    previous_scores: List[float]
    baseline_score: float


@dataclass
class PerformanceTrackingOutput:
    current_score: float
    improvement: float
    should_continue: bool
    reason: str
    all_scores: List[float]


class PerformanceTrackingAgent(Agent):
    """Agent responsible for tracking performance improvements and deciding continuation."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def run(self, input: AgentInput) -> AgentOutput:
        # Handle input from previous agent or direct call
        if hasattr(input.data, 'pruned_df'):
            # Input is a FeaturePruningOutput object
            df = input.data.pruned_df
            target_column = self.config.get("target_column", "class")
            n_folds = self.config.get("n_folds", 10)
            test_size = self.config.get("test_size", 0.2)
            model = self.config.get("model", "tabpfn")
            # We need to get other parameters from somewhere - this is a limitation
            # For now, we'll need to pass them differently
            current_iteration = 0
            max_iterations = 10
            previous_scores = []
            baseline_score = 0.0
        else:
            # Input is a dictionary (direct call)
            tracking_input = PerformanceTrackingInput(**input.data)
            df = tracking_input.df
            target_column = tracking_input.target_column
            n_folds = tracking_input.n_folds
            test_size = tracking_input.test_size
            model = tracking_input.model
            current_iteration = tracking_input.current_iteration
            max_iterations = tracking_input.max_iterations
            previous_scores = tracking_input.previous_scores
            baseline_score = tracking_input.baseline_score
        
        # Create evaluator
        evaluator = EvaluationAgent(
            label=target_column,
            n_folds=n_folds,
            test_size=test_size,
            model=model
        )
        
        # Evaluate current performance using nested cross-validation
        eval_input = AgentInput(data={
            "df": tracking_input.df,
            "target_column": tracking_input.target_column,
            "evaluation_type": "nested_cv",
            "n_splits": tracking_input.n_folds
        })
        eval_output = evaluator.run(eval_input)
        current_scores = eval_output.result.scores
        current_score = eval_output.result.mean_score
        
        # Calculate improvement
        improvement = current_score - tracking_input.baseline_score
        
        # Determine if we should continue
        should_continue = True
        reason = "Performance tracking continues"
        
        # Check iteration limit
        if tracking_input.current_iteration >= tracking_input.max_iterations:
            should_continue = False
            reason = f"Reached maximum iterations ({tracking_input.max_iterations})"
        
        # Check for perfect score
        elif current_score > 0.9999999:
            should_continue = False
            reason = "Perfect score achieved"
        
        # Check if performance improved from previous iteration
        elif tracking_input.previous_scores:
            best_previous = max(tracking_input.previous_scores)
            if current_score <= best_previous:
                should_continue = False
                reason = f"No improvement: current ({current_score:.4f}) <= best previous ({best_previous:.4f})"
        
        # Update scores list
        all_scores = tracking_input.previous_scores + [current_score]
        
        logger.debug(f"Current score: {current_score:.4f}")
        logger.debug(f"Improvement from baseline: {improvement:.4f}")
        logger.debug(f"Should continue: {should_continue} - {reason}")
        
        output = PerformanceTrackingOutput(
            current_score=current_score,
            improvement=improvement,
            should_continue=should_continue,
            reason=reason,
            all_scores=all_scores
        )
        
        return AgentOutput(
            result=output,
            metadata={"agent": "PerformanceTrackingAgent", "status": "success"}
        )
