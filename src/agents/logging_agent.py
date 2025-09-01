from dataclasses import dataclass
from typing import Dict, Any, List
import os
from datetime import datetime
import json
import numpy as np

from src.agents.core.agent import Agent, AgentInput, AgentOutput
from src.utils.funcs import write_to_logs
from src import logger


@dataclass
class LoggingInput:
    iteration: int
    domain_prompt: str
    domain_context: Dict
    augment_prompt: str
    augment_response: List
    baseline_score: float
    scores_before_pruning: List[float]
    scores_after_pruning: List[float]
    success: bool = True
    error_message: str = ""


@dataclass
class LoggingOutput:
    log_file_path: str
    log_entries: int
    success: bool


class LoggingAgent(Agent):
    """Agent responsible for logging and monitoring the TDA process."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_entries = 0
    
    def run(self, input: AgentInput) -> AgentOutput:
        logging_input = LoggingInput(**input.data)
        
        try:
            # Create timestamp for this log entry
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Write to logs using existing utility
            write_to_logs(
                logging_input.domain_prompt,
                logging_input.domain_context,
                logging_input.augment_prompt,
                logging_input.augment_response,
                logging_input.baseline_score,
                logging_input.scores_before_pruning,
                logging_input.scores_after_pruning,
            )
            
            # Create additional structured log entry
            log_entry = {
                "timestamp": timestamp,
                "iteration": logging_input.iteration,
                "baseline_score": logging_input.baseline_score,
                "score_before_pruning": np.mean(logging_input.scores_before_pruning) if logging_input.scores_before_pruning else None,
                "score_after_pruning": np.mean(logging_input.scores_after_pruning) if logging_input.scores_after_pruning else None,
                "success": logging_input.success,
                "error_message": logging_input.error_message,
                "domain_context_keys": list(logging_input.domain_context.keys()) if logging_input.domain_context else [],
                "augment_response_count": len(logging_input.augment_response) if logging_input.augment_response else 0
            }
            
            # Save structured log
            log_file_path = os.path.join(self.log_dir, f"tda_iteration_{logging_input.iteration:03d}_{timestamp}.json")
            with open(log_file_path, 'w') as f:
                json.dump(log_entry, f, indent=2, default=str)
            
            self.log_entries += 1
            
            output = LoggingOutput(
                log_file_path=log_file_path,
                log_entries=self.log_entries,
                success=True
            )
            
            logger.debug(f"Logging completed for iteration {logging_input.iteration}")
            
            return AgentOutput(
                result=output,
                metadata={"agent": "LoggingAgent", "status": "success"}
            )
            
        except Exception as e:
            logger.error(f"Logging failed: {str(e)}")
            
            output = LoggingOutput(
                log_file_path="",
                log_entries=self.log_entries,
                success=False
            )
            
            return AgentOutput(
                result=output,
                metadata={"agent": "LoggingAgent", "status": "failed", "error": str(e)}
            )
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about logging."""
        log_files = [f for f in os.listdir(self.log_dir) if f.endswith('.json')]
        
        return {
            "total_log_entries": self.log_entries,
            "log_files_count": len(log_files),
            "log_directory": self.log_dir,
            "log_files": log_files
        }
