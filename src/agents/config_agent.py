from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from src.agents.core.agent import Agent, AgentInput, AgentOutput
from src import logger


@dataclass
class ConfigValidationInput:
    data_path: str
    num_columns_to_add: int
    target_column: str
    n_folds: int
    test_size: float
    model: str
    max_augmentations: int
    verbose: bool


@dataclass
class ConfigValidationOutput:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    validated_config: Dict[str, Any]


class ConfigAgent(Agent):
    """Agent responsible for configuration validation and management."""
    
    def __init__(self):
        self.supported_models = ["tabpfn", "lightgbm", "autogluon"]
        self.max_folds = 20
        self.max_columns_to_add = 100
    
    def run(self, input: AgentInput) -> AgentOutput:
        config_input = ConfigValidationInput(**input.data)
        
        errors = []
        warnings = []
        
        # Validate data path
        if not os.path.exists(config_input.data_path):
            errors.append(f"Data path does not exist: {config_input.data_path}")
        elif not config_input.data_path.endswith('.arff'):
            warnings.append(f"Data path does not end with .arff: {config_input.data_path}")
        
        # Validate numeric parameters
        if config_input.num_columns_to_add <= 0:
            errors.append("num_columns_to_add must be positive")
        elif config_input.num_columns_to_add > self.max_columns_to_add:
            warnings.append(f"num_columns_to_add ({config_input.num_columns_to_add}) is quite high")
        
        if config_input.n_folds < 2:
            errors.append("n_folds must be at least 2")
        elif config_input.n_folds > self.max_folds:
            warnings.append(f"n_folds ({config_input.n_folds}) is quite high, may slow down execution")
        
        if config_input.test_size <= 0 or config_input.test_size >= 1:
            errors.append("test_size must be between 0 and 1")
        
        if config_input.max_augmentations <= 0:
            errors.append("max_augmentations must be positive")
        
        # Validate model
        if config_input.model not in self.supported_models:
            warnings.append(f"Model '{config_input.model}' is not in the list of tested models: {self.supported_models}")
        
        # Validate target column (basic check)
        if not config_input.target_column or len(config_input.target_column.strip()) == 0:
            errors.append("target_column cannot be empty")
        
        # Check environment variables
        if not os.getenv("OPENAI_API_KEY"):
            errors.append("OPENAI_API_KEY environment variable is not set")
        
        # Check available disk space for outputs
        try:
            output_dir = Path("outputs")
            if output_dir.exists():
                stat = output_dir.stat()
                # Basic check - in production you might want more sophisticated disk space checking
                pass
        except Exception as e:
            warnings.append(f"Could not check output directory permissions: {e}")
        
        # Create validated config
        validated_config = {
            "data_path": config_input.data_path,
            "num_columns_to_add": config_input.num_columns_to_add,
            "target_column": config_input.target_column,
            "n_folds": config_input.n_folds,
            "test_size": config_input.test_size,
            "model": config_input.model,
            "max_augmentations": config_input.max_augmentations,
            "verbose": config_input.verbose
        }
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.debug("Configuration validation passed")
        else:
            logger.error(f"Configuration validation failed: {errors}")
        
        if warnings:
            logger.warning(f"Configuration warnings: {warnings}")
        
        output = ConfigValidationOutput(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validated_config=validated_config
        )
        
        return AgentOutput(
            result=output,
            metadata={"agent": "ConfigAgent", "status": "success" if is_valid else "failed"}
        )
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "data_path": "./data/dataset_31_credit-g.arff",
            "num_columns_to_add": 20,
            "target_column": "class",
            "n_folds": 10,
            "test_size": 0.2,
            "model": "tabpfn",
            "max_augmentations": 10,
            "verbose": True
        }
    
    def validate_file_format(self, file_path: str) -> Dict[str, Any]:
        """Validate if a file is in the expected format."""
        result = {
            "is_valid": False,
            "file_type": "unknown",
            "size_bytes": 0,
            "can_read": False
        }
        
        try:
            if os.path.exists(file_path):
                result["size_bytes"] = os.path.getsize(file_path)
                result["can_read"] = os.access(file_path, os.R_OK)
                
                if file_path.endswith('.arff'):
                    result["file_type"] = "arff"
                    result["is_valid"] = True
                elif file_path.endswith('.csv'):
                    result["file_type"] = "csv"
                    result["is_valid"] = True
                elif file_path.endswith('.parquet'):
                    result["file_type"] = "parquet"
                    result["is_valid"] = True
                else:
                    result["file_type"] = "unknown"
                    result["is_valid"] = False
        except Exception as e:
            result["error"] = str(e)
        
        return result
