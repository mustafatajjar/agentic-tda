from dataclasses import dataclass
from typing import Dict, Any, List
import os
import numpy as np
import pandas as pd

from src.agents.core.agent_pipeline import RoutePipeline
from src.agents.core.agent import Agent, AgentInput, AgentOutput
from src.agents.data_preparation_agent import DataPreparationAgent
from src.agents.baseline_evaluation_agent import BaselineEvaluationAgent
from src.agents.domain_agent import DomainAgent
from src.agents.augment import AugmentAgent
from src.agents.feature_pruning_agent import FeaturePruningAgent
from src.agents.performance_tracking_agent import PerformanceTrackingAgent
from src.agents.eval_agent import EvaluationAgent
from src.utils.funcs import write_to_logs
from src import logger


@dataclass
class TDAPipelineConfig:
    data_path: str
    num_columns_to_add: int = 20
    target_column: str = "class"
    n_folds: int = 10
    test_size: float = 0.2
    model: str = "tabpfn"
    max_augmentations: int = 10
    verbose: bool = True


class TDAPipeline:
    """
    Main pipeline orchestrating the TDA process using a route-based approach.
    
    NOTE: This pipeline has data flow issues between agents due to the RoutePipeline
    architecture not properly handling the data types expected by each agent.
    Use SimplifiedTDAPipeline instead for now.
    """
    
    def __init__(self, config: TDAPipelineConfig):
        self.config = config
        self.df_all_augmented = None
        self.augment_responses = []
        self.selected_features_history = []
        self.current_iteration = 0
        
        # Initialize agents
        self.agents = {
            "data_prep": DataPreparationAgent(),
            "baseline_eval": BaselineEvaluationAgent({
                "target_column": self.config.target_column,
                "n_folds": self.config.n_folds,
                "test_size": self.config.test_size,
                "model": self.config.model
            }),
            "domain_analysis": DomainAgent(),
            "augment": AugmentAgent(),
            "prune": FeaturePruningAgent(),
            "performance_track": PerformanceTrackingAgent({
                "target_column": self.config.target_column,
                "n_folds": self.config.n_folds,
                "test_size": self.config.test_size,
                "model": self.config.model
            }),
            "final_eval": EvaluationAgent(
                label=self.config.target_column,
                n_folds=self.config.n_folds,
                test_size=self.config.test_size,
                model=self.config.model
            )
        }
        
        # Define the pipeline flow
        self.routes = {
            "data_prep": self._route_from_data_prep,
            "baseline_eval": self._route_from_baseline_eval,
            "domain_analysis": self._route_from_domain_analysis,
            "augment": self._route_from_augment,
            "prune": self._route_from_prune,
            "performance_track": self._route_from_performance_track,
        }
        
        # Create the route pipeline
        self.pipeline = RoutePipeline(
            agents=self.agents,
            routes=self.routes,
            start="data_prep",
            max_steps=100  # Safety limit
        )
    
    def run(self) -> Dict[str, Any]:
        """Run the complete TDA pipeline."""
        logger.debug(f"Starting TDA pipeline with config: {self.config}")
        
        # Initial input
        initial_input = AgentInput(
            data={
                "data_path": self.config.data_path,
                "n_folds": self.config.n_folds
            }
        )
        
        # Run the pipeline
        final_output = self.pipeline.run(initial_input)
        
        # Save final results
        if self.df_all_augmented is not None:
            self.df_all_augmented.to_csv("all_augmented_columns.csv", index=False)
            logger.debug("Saved all augmented columns to all_augmented_columns.csv")
        
        return {
            "final_output": final_output,
            "total_iterations": self.current_iteration,
            "augment_responses": self.augment_responses,
            "selected_features_history": self.selected_features_history
        }
    
    def _route_from_data_prep(self, output: AgentOutput) -> str:
        """Route after data preparation."""
        logger.debug("Data preparation completed, moving to baseline evaluation")
        # Store the data preparation output for use in baseline evaluation
        self.data_prep_output = output.result
        return "baseline_eval"
    
    def _route_from_baseline_eval(self, output: AgentOutput) -> str:
        """Route after baseline evaluation."""
        logger.debug("Baseline evaluation completed, moving to domain analysis")
        # Store the baseline evaluation output
        self.baseline_output = output.result
        return "domain_analysis"
    
    def _route_from_domain_analysis(self, output: AgentOutput) -> str:
        """Route after domain analysis."""
        logger.debug("Domain analysis completed, starting augmentation loop")
        # Store the domain analysis output
        self.domain_output = output.result
        return "augment"
    
    def _route_from_augment(self, output: AgentOutput) -> str:
        """Route after augmentation."""
        if not output.result["success"]:
            logger.debug("Augmentation failed, stopping")
            return None
        
        # Update tracking data
        self.augment_responses.append({"response": output.result["suggestions"]})
        
        # Add new columns to tracking dataframe
        if self.df_all_augmented is None:
            # First iteration - initialize from baseline
            if hasattr(self, 'data_prep_output'):
                self.df_all_augmented = self.data_prep_output.df.copy()
        
        # Add new columns
        augmented_df = output.result["augmented_df"]
        for col in augmented_df.columns:
            if col not in self.df_all_augmented.columns:
                self.df_all_augmented[col] = augmented_df[col]
        
        logger.debug("Augmentation completed, moving to feature pruning")
        return "prune"
    
    def _route_from_prune(self, output: AgentOutput) -> str:
        """Route after feature pruning."""
        logger.debug("Feature pruning completed, moving to performance tracking")
        return "performance_track"
    
    def _route_from_performance_track(self, output: AgentOutput) -> str:
        """Route after performance tracking."""
        self.current_iteration += 1
        
        if not output.result["should_continue"]:
            logger.debug(f"Stopping augmentation: {output.result['reason']}")
            return None
        
        # Continue with next iteration
        logger.debug(f"Continuing to iteration {self.current_iteration + 1}")
        return "augment"
    
    def _get_agent_output(self, agent_name: str) -> AgentOutput:
        """Helper to get output from a specific agent (for cross-referencing)."""
        # This is a simplified approach - in a real implementation,
        # you might want to store agent outputs in a more sophisticated way
        return None


class SimplifiedTDAPipeline:
    """A simplified version that maintains the original logic but with better structure."""
    
    def __init__(self, config: TDAPipelineConfig):
        self.config = config
        self.df_all_augmented = None
        self.augment_responses = []
        self.selected_features_history = []
        
        # Initialize agents
        self.data_prep_agent = DataPreparationAgent()
        self.baseline_eval_agent = BaselineEvaluationAgent()
        self.domain_agent = DomainAgent()
        self.augment_agent = AugmentAgent()
        self.prune_agent = FeaturePruningAgent()
        self.performance_track_agent = PerformanceTrackingAgent()
        self.evaluator = EvaluationAgent(
            label=self.config.target_column,
            n_folds=self.config.n_folds,
            test_size=self.config.test_size,
            model=self.config.model
        )
    
    def run(self) -> Dict[str, Any]:
        """Run the simplified TDA pipeline."""
        logger.debug(f"Starting simplified TDA pipeline with config: {self.config}")
        
        # Step 1: Data Preparation
        data_prep_input = AgentInput(data={
            "data_path": self.config.data_path,
            "n_folds": self.config.n_folds
        })
        data_prep_output = self.data_prep_agent.run(data_prep_input)
        df = data_prep_output.result.df
        metadata = data_prep_output.result.metadata
        
        # Step 2: Baseline Evaluation
        baseline_input = AgentInput(data={
            "df": df,
            "target_column": self.config.target_column,
            "n_folds": self.config.n_folds,
            "test_size": self.config.test_size,
            "model": self.config.model
        })
        baseline_output = self.baseline_eval_agent.run(baseline_input)
        baseline_score = baseline_output.result.baseline_score
        evals = [baseline_score]
        
        # Initialize tracking dataframe
        self.df_all_augmented = df.copy()
        
        # Step 3: Main augmentation loop
        iteration = 0
        while iteration < self.config.max_augmentations:
            logger.debug(f"Starting iteration {iteration + 1}")
            
            # Domain Analysis
            domain_input = AgentInput(data={
                "df": df,
                "arff_metadata": metadata
            })
            domain_output = self.domain_agent.run(domain_input)
            domain_context = domain_output.result
            
            # Augmentation
            augment_input = AgentInput(data={
                "df": df.copy(),
                "domain_context": domain_context,
                "history_responses": self.augment_responses,
                "selected_features_history": self.selected_features_history,
                "num_columns_to_add": self.config.num_columns_to_add,
                "target_column": self.config.target_column
            })
            augment_output = self.augment_agent.run(augment_input)
            
            if not augment_output.result["success"]:
                logger.debug("Augmentation failed, stopping")
                break
            
            # Update tracking
            self.augment_responses.append({"response": augment_output.result["suggestions"]})
            
            # Add new columns to tracking dataframe
            augmented_df = augment_output.result["augmented_df"]
            for col in augmented_df.columns:
                if col not in self.df_all_augmented.columns:
                    self.df_all_augmented[col] = augmented_df[col]
            
            # Feature Pruning
            prune_input = AgentInput(data={
                "df": augmented_df,
                "target_column": self.config.target_column
            })
            prune_output = self.prune_agent.run(prune_input)
            pruned_df = prune_output.result.pruned_df
            
            # Performance Tracking
            track_input = AgentInput(data={
                "df": pruned_df,
                "target_column": self.config.target_column,
                "n_folds": self.config.n_folds,
                "test_size": self.config.test_size,
                "model": self.config.model,
                "current_iteration": iteration,
                "max_iterations": self.config.max_augmentations,
                "previous_scores": evals,
                "baseline_score": baseline_score
            })
            track_output = self.performance_track_agent.run(track_input)
            
            # Update evaluation scores
            current_score = track_output.result.current_score
            evals.append(current_score)
            
            # Log results if verbose
            if self.config.verbose:
                write_to_logs(
                    domain_output.metadata.get("prompt", ""),
                    domain_context,
                    augment_output.metadata.get("prompt", ""),
                    augment_output.result["suggestions"],
                    baseline_score,
                    [current_score],  # Before pruning
                    [current_score],  # After pruning
                )
            
            # Check if we should continue
            if not track_output.result.should_continue:
                logger.debug(f"Stopping: {track_output.result.reason}")
                break
            
            # Update dataframe for next iteration
            df = pruned_df.copy()
            iteration += 1
        
        # Final evaluation
        final_eval_input = AgentInput(data={
            "df": df,
            "target_column": self.config.target_column,
            "evaluation_type": "kfold",
            "n_splits": self.config.n_folds,
            "device": "cuda"
        })
        final_eval_output = self.evaluator.run(final_eval_input)
        final_eval = final_eval_output.result.mean_score
        
        logger.debug(f"Final holdout evaluation: {final_eval}")
        logger.debug(f"Baseline score: {baseline_score}")
        
        # Save results
        self.df_all_augmented.to_csv("all_augmented_columns.csv", index=False)
        logger.debug("Saved all augmented columns to all_augmented_columns.csv")
        
        return {
            "final_score": final_eval,
            "baseline_score": baseline_score,
            "total_iterations": iteration,
            "augment_responses": self.augment_responses,
            "selected_features_history": self.selected_features_history,
            "evaluation_history": evals
        }
