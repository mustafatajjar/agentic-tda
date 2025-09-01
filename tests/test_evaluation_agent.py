import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import tempfile
import os

from src.agents.eval_agent import EvaluationAgent, EvaluationInput, EvaluationOutput
from src.agents.core.agent import AgentInput, AgentOutput


class TestEvaluationAgent(unittest.TestCase):
    """Test cases for the refactored EvaluationAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test dataset
        self.test_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'class': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Create evaluation agent
        self.agent = EvaluationAgent(
            label="class",
            n_folds=3,
            test_size=0.3,
            model="lightgbm"
        )
    
    def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.label, "class")
        self.assertEqual(self.agent.n_folds, 3)
        self.assertEqual(self.agent.test_size, 0.3)
        self.assertEqual(self.agent.model_type, "lightgbm")
        self.assertIsNone(self.agent.train_indices)
        self.assertIsNone(self.agent.test_indices)
        self.assertIsNone(self.agent.fold_indices)
    
    def test_model_initialization_lightgbm(self):
        """Test LightGBM model initialization."""
        agent = EvaluationAgent(model="lightgbm")
        self.assertEqual(agent.model_type, "lightgbm")
        self.assertFalse(agent.use_values)
    
    def test_model_initialization_tabpfn(self):
        """Test TabPFN model initialization."""
        agent = EvaluationAgent(model="tabpfn")
        self.assertEqual(agent.model_type, "tabpfn")
        self.assertTrue(agent.use_values)
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with self.assertRaises(ValueError):
            EvaluationAgent(model="invalid_model")
    
    def test_run_method_nested_cv(self):
        """Test the run method with nested cross-validation."""
        input_data = AgentInput(data={
            "df": self.test_df,
            "target_column": "class",
            "evaluation_type": "nested_cv",
            "n_splits": 3
        })
        
        # Mock the model to avoid actual training
        with patch.object(self.agent.model, 'fit') as mock_fit:
            with patch.object(self.agent.model, 'predict_proba') as mock_predict:
                mock_predict.return_value = np.array([[0.3, 0.7], [0.2, 0.8], [0.4, 0.6]])
                
                output = self.agent.run(input_data)
                
                self.assertIsInstance(output, AgentOutput)
                self.assertEqual(output.result.evaluation_type, "nested_cv")
                self.assertEqual(output.result.n_splits, 3)
                self.assertIsInstance(output.result.scores, list)
                self.assertIsInstance(output.result.mean_score, float)
                self.assertIsInstance(output.result.std_score, float)
    
    def test_run_method_holdout(self):
        """Test the run method with holdout evaluation."""
        input_data = AgentInput(data={
            "df": self.test_df,
            "target_column": "class",
            "evaluation_type": "holdout",
            "n_splits": 1
        })
        
        # Mock the model to avoid actual training
        with patch.object(self.agent.model, 'fit') as mock_fit:
            with patch.object(self.agent.model, 'predict_proba') as mock_predict:
                mock_predict.return_value = np.array([[0.3, 0.7]])
                
                output = self.agent.run(input_data)
                
                self.assertIsInstance(output, AgentOutput)
                self.assertEqual(output.result.evaluation_type, "holdout")
                self.assertEqual(output.result.n_splits, 1)
                self.assertIsInstance(output.result.scores, list)
                self.assertEqual(len(output.result.scores), 1)
    
    def test_run_method_kfold(self):
        """Test the run method with k-fold cross-validation."""
        input_data = AgentInput(data={
            "df": self.test_df,
            "target_column": "class",
            "evaluation_type": "kfold",
            "n_splits": 3
        })
        
        # Mock the model to avoid actual training
        with patch.object(self.agent.model, 'fit') as mock_fit:
            with patch.object(self.agent.model, 'predict_proba') as mock_predict:
                mock_predict.return_value = np.array([[0.3, 0.7], [0.2, 0.8], [0.4, 0.6]])
                
                output = self.agent.run(input_data)
                
                self.assertIsInstance(output, AgentOutput)
                self.assertEqual(output.result.evaluation_type, "kfold")
                self.assertEqual(output.result.n_splits, 3)
                self.assertIsInstance(output.result.scores, list)
                self.assertEqual(len(output.result.scores), 3)
    
    def test_invalid_evaluation_type(self):
        """Test that invalid evaluation type raises error."""
        input_data = AgentInput(data={
            "df": self.test_df,
            "target_column": "class",
            "evaluation_type": "invalid_type",
            "n_splits": 3
        })
        
        with self.assertRaises(ValueError):
            self.agent.run(input_data)
    
    def test_indices_setup(self):
        """Test that indices are set up correctly."""
        input_data = AgentInput(data={
            "df": self.test_df,
            "target_column": "class",
            "evaluation_type": "nested_cv"
        })
        
        # Run once to set up indices
        with patch.object(self.agent.model, 'fit'):
            with patch.object(self.agent.model, 'predict_proba') as mock_predict:
                mock_predict.return_value = np.array([[0.3, 0.7], [0.2, 0.8], [0.4, 0.6]])
                self.agent.run(input_data)
        
        # Check that indices are now set
        self.assertIsNotNone(self.agent.train_indices)
        self.assertIsNotNone(self.agent.test_indices)
        self.assertIsNotNone(self.agent.fold_indices)
        
        # Check that we have the right number of folds
        self.assertEqual(len(self.agent.fold_indices), 3)
    
    def test_backward_compatibility_methods(self):
        """Test that backward compatibility methods still work."""
        # Test nested_cross_val
        scores = self.agent.nested_cross_val(self.test_df)
        self.assertIsInstance(scores, list)
        
        # Test test_on_holdout_kfold
        scores = self.agent.test_on_holdout_kfold(self.test_df, n_splits=3)
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 3)
        
        # Test test_on_holdout
        score = self.agent.test_on_holdout(self.test_df)
        self.assertIsInstance(score, float)
    
    def test_get_indices_methods(self):
        """Test the get_indices methods."""
        # Should raise error before indices are set up
        with self.assertRaises(ValueError):
            self.agent.get_fold_indices()
        
        with self.assertRaises(ValueError):
            self.agent.get_train_test_indices()
        
        # Set up indices
        input_data = AgentInput(data={
            "df": self.test_df,
            "target_column": "class",
            "evaluation_type": "nested_cv"
        })
        
        with patch.object(self.agent.model, 'fit'):
            with patch.object(self.agent.model, 'predict_proba') as mock_predict:
                mock_predict.return_value = np.array([[0.3, 0.7], [0.2, 0.8], [0.4, 0.6]])
                self.agent.run(input_data)
        
        # Now should work
        fold_indices = self.agent.get_fold_indices()
        self.assertIsInstance(fold_indices, list)
        
        train_indices, test_indices = self.agent.get_train_test_indices()
        self.assertIsInstance(train_indices, np.ndarray)
        self.assertIsInstance(test_indices, np.ndarray)
    
    def test_metadata_output(self):
        """Test that metadata is correctly set in output."""
        input_data = AgentInput(data={
            "df": self.test_df,
            "target_column": "class",
            "evaluation_type": "nested_cv"
        })
        
        with patch.object(self.agent.model, 'fit'):
            with patch.object(self.agent.model, 'predict_proba') as mock_predict:
                mock_predict.return_value = np.array([[0.3, 0.7], [0.2, 0.8], [0.4, 0.6]])
                
                output = self.agent.run(input_data)
                
                self.assertIn("agent", output.metadata)
                self.assertIn("model_type", output.metadata)
                self.assertIn("status", output.metadata)
                self.assertEqual(output.metadata["agent"], "EvaluationAgent")
                self.assertEqual(output.metadata["model_type"], "lightgbm")
                self.assertEqual(output.metadata["status"], "success")


if __name__ == '__main__':
    unittest.main()
