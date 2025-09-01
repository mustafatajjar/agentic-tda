import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import tempfile
import os

from src.agents.tda_pipeline import TDAPipeline, TDAPipelineConfig, SimplifiedTDAPipeline
from src.agents.config_agent import ConfigAgent
from src.agents.data_preparation_agent import DataPreparationAgent
from src.agents.baseline_evaluation_agent import BaselineEvaluationAgent
from src.agents.domain_agent import DomainAgent
from src.agents.augment import AugmentAgent
from src.agents.feature_pruning_agent import FeaturePruningAgent
from src.agents.performance_tracking_agent import PerformanceTrackingAgent
from src.agents.logging_agent import LoggingAgent
from src.agents.core.agent import AgentInput, AgentOutput


class TestTDAPipeline(unittest.TestCase):
    """Test cases for the new TDA pipeline architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TDAPipelineConfig(
            data_path="./data/test_dataset.arff",
            num_columns_to_add=5,
            target_column="class",
            n_folds=5,
            test_size=0.2,
            model="tabpfn",
            max_augmentations=3,
            verbose=False
        )
        
        # Create a temporary test dataset
        self.test_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'class': [0, 1, 0, 1, 0]
        })
    
    def test_config_agent_validation(self):
        """Test configuration validation."""
        config_agent = ConfigAgent()
        
        # Test valid config
        valid_input = AgentInput(data={
            "data_path": "./data/test.arff",
            "num_columns_to_add": 10,
            "target_column": "class",
            "n_folds": 5,
            "test_size": 0.2,
            "model": "tabpfn",
            "max_augmentations": 5,
            "verbose": True
        })
        
        with patch('os.path.exists', return_value=True):
            with patch('os.getenv', return_value="test_key"):
                output = config_agent.run(valid_input)
                self.assertTrue(output.result.is_valid)
                self.assertEqual(len(output.result.errors), 0)
    
    def test_data_preparation_agent(self):
        """Test data preparation agent."""
        agent = DataPreparationAgent()
        
        # Mock the utility functions
        with patch('src.utils.funcs.arff_to_dataframe') as mock_arff:
            with patch('src.utils.funcs.extract_arff_metadata') as mock_metadata:
                mock_arff.return_value = self.test_df
                mock_metadata.return_value = "Test metadata"
                
                input_data = AgentInput(data={
                    "data_path": "./data/test.arff",
                    "n_folds": 5
                })
                
                output = agent.run(input_data)
                
                self.assertIsNotNone(output.result.df)
                self.assertEqual(output.result.metadata, "Test metadata")
                self.assertEqual(len(output.result.fold_indices), 5)
    
    def test_baseline_evaluation_agent(self):
        """Test baseline evaluation agent."""
        agent = BaselineEvaluationAgent()
        
        # Mock the evaluation agent
        with patch('src.agents.eval_agent.EvaluationAgent') as mock_eval:
            mock_instance = Mock()
            mock_instance.test_on_holdout_kfold.return_value = [0.8, 0.85]
            mock_instance.nested_cross_val.return_value = [0.75, 0.8, 0.85]
            mock_eval.return_value = mock_instance
            
            input_data = AgentInput(data={
                "df": self.test_df,
                "target_column": "class",
                "n_folds": 5,
                "test_size": 0.2,
                "model": "tabpfn"
            })
            
            output = agent.run(input_data)
            
            self.assertIsNotNone(output.result.original_eval)
            self.assertIsNotNone(output.result.baseline_score)
            self.assertEqual(len(output.result.original_nested_cv_scores), 3)
    
    def test_domain_agent(self):
        """Test domain agent."""
        agent = DomainAgent()
        
        # Mock OpenAI client
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"primary_domain": "test", "column_descriptions": {}, "important_metadata": ""}'
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Mock the summarize_dataframe function
            with patch('src.utils.funcs.summarize_dataframe') as mock_summary:
                mock_summary.return_value = pd.DataFrame({'col': ['val']})
                
                input_data = AgentInput(data={
                    "df": self.test_df,
                    "arff_metadata": "Test metadata"
                })
                
                output = agent.run(input_data)
                
                self.assertIsNotNone(output.result)
                self.assertIn("primary_domain", output.result)
    
    def test_augment_agent(self):
        """Test augmentation agent."""
        agent = AugmentAgent()
        
        # Mock OpenAI client
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"columns": [{"name": "new_feature", "generation_method": "df[\'new_feature\'] = df[\'feature1\'] * 2"}]}'
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Mock the summarize_dataframe function
            with patch('src.utils.funcs.summarize_dataframe') as mock_summary:
                mock_summary.return_value = pd.DataFrame({'col': ['val']})
                
                input_data = AgentInput(data={
                    "df": self.test_df,
                    "domain_context": {"primary_domain": "test"},
                    "history_responses": [],
                    "selected_features_history": [],
                    "num_columns_to_add": 1,
                    "target_column": "class"
                })
                
                output = agent.run(input_data)
                
                self.assertIsNotNone(output.result)
                self.assertTrue(output.result["success"])
                self.assertIn("augmented_df", output.result)
    
    def test_feature_pruning_agent(self):
        """Test feature pruning agent."""
        agent = FeaturePruningAgent()
        
        # Mock the prune_features_binary_classification function
        with patch('src.utils.funcs.prune_features_binary_classification') as mock_prune:
            mock_prune.return_value = [['feature1'], ['feature1'], ['feature1']]
            
            input_data = AgentInput(data={
                "df": self.test_df,
                "target_column": "class"
            })
            
            output = agent.run(input_data)
            
            self.assertIsNotNone(output.result.pruned_df)
            self.assertIsNotNone(output.result.selected_features)
            self.assertTrue(output.result.pruning_effective)
    
    def test_performance_tracking_agent(self):
        """Test performance tracking agent."""
        agent = PerformanceTrackingAgent()
        
        # Mock the evaluation agent
        with patch('src.agents.eval_agent.EvaluationAgent') as mock_eval:
            mock_instance = Mock()
            mock_instance.nested_cross_val.return_value = [0.8, 0.85]
            mock_eval.return_value = mock_instance
            
            input_data = AgentInput(data={
                "df": self.test_df,
                "target_column": "class",
                "n_folds": 5,
                "test_size": 0.2,
                "model": "tabpfn",
                "current_iteration": 1,
                "max_iterations": 5,
                "previous_scores": [0.75],
                "baseline_score": 0.7
            })
            
            output = agent.run(input_data)
            
            self.assertIsNotNone(output.result.current_score)
            self.assertIsNotNone(output.result.should_continue)
            self.assertIsNotNone(output.result.reason)
    
    def test_simplified_pipeline_creation(self):
        """Test simplified pipeline creation."""
        pipeline = SimplifiedTDAPipeline(self.config)
        
        self.assertIsNotNone(pipeline.data_prep_agent)
        self.assertIsNotNone(pipeline.baseline_eval_agent)
        self.assertIsNotNone(pipeline.domain_agent)
        self.assertIsNotNone(pipeline.augment_agent)
        self.assertIsNotNone(pipeline.prune_agent)
        self.assertIsNotNone(pipeline.performance_track_agent)
    
    def test_pipeline_config_validation(self):
        """Test pipeline configuration validation."""
        # Test invalid config
        invalid_config = TDAPipelineConfig(
            data_path="",  # Invalid path
            num_columns_to_add=-1,  # Invalid number
            target_column="",  # Invalid target
            n_folds=1,  # Invalid folds
            test_size=1.5,  # Invalid test size
            model="invalid_model",  # Invalid model
            max_augmentations=0,  # Invalid max
            verbose=True
        )
        
        # This should raise an error or handle gracefully
        # The exact behavior depends on your validation logic
        pass
    
    def test_logging_agent(self):
        """Test logging agent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = LoggingAgent(log_dir=temp_dir)
            
            input_data = AgentInput(data={
                "iteration": 1,
                "domain_prompt": "Test prompt",
                "domain_context": {"key": "value"},
                "augment_prompt": "Test augment prompt",
                "augment_response": [{"test": "response"}],
                "baseline_score": 0.7,
                "scores_before_pruning": [0.8],
                "scores_after_pruning": [0.85],
                "success": True
            })
            
            # Mock the write_to_logs function
            with patch('src.utils.funcs.write_to_logs'):
                output = agent.run(input_data)
                
                self.assertTrue(output.result.success)
                self.assertIsNotNone(output.result.log_file_path)
    
    def test_pipeline_integration(self):
        """Test basic pipeline integration."""
        # This is a high-level test to ensure the pipeline can be created
        # and basic components work together
        pipeline = SimplifiedTDAPipeline(self.config)
        
        # Test that all required methods exist
        self.assertTrue(hasattr(pipeline, 'run'))
        self.assertTrue(hasattr(pipeline, 'config'))
        
        # Test configuration access
        self.assertEqual(pipeline.config.data_path, self.config.data_path)
        self.assertEqual(pipeline.config.num_columns_to_add, self.config.num_columns_to_add)


if __name__ == '__main__':
    unittest.main()
