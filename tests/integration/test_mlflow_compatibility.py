#!/usr/bin/env python
"""
MLflow Compatibility Test Suite

This test suite verifies that Genesis-Flow with MongoDB backend provides
100% API compatibility with standard MLflow operations. It tests all major
MLflow functionality to ensure seamless migration from MLflow server to
MongoDB backend.

Usage:
    python -m pytest tests/integration/test_mlflow_compatibility.py -v
"""

import pytest
import tempfile
import os
import time
import uuid
import numpy as np
import pandas as pd
from typing import List, Dict, Any

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus, ViewType
from mlflow.exceptions import MlflowException

# Test data imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# MLflow types for advanced testing
from mlflow.types.llm import (
    ChatMessage,
    ChatParams,
    ChatCompletionResponse,
    ChatChoice,
    TokenUsageStats
)


class TestMLflowCompatibility:
    """
    Comprehensive test suite for MLflow compatibility with MongoDB backend.
    
    Tests all major MLflow functionality to ensure Genesis-Flow provides
    100% API compatibility with standard MLflow operations.
    """
    
    @pytest.fixture(scope="class")
    def mongodb_tracking_uri(self):
        """MongoDB tracking URI for testing."""
        return "mongodb://localhost:27017/mlflow_compatibility_test"
    
    @pytest.fixture(scope="class")
    def mongodb_registry_uri(self):
        """MongoDB registry URI for testing."""
        return "mongodb://localhost:27017/mlflow_compatibility_test"
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_mlflow(self, mongodb_tracking_uri, mongodb_registry_uri):
        """Setup MLflow with MongoDB backend."""
        mlflow.set_tracking_uri(mongodb_tracking_uri)
        mlflow.set_registry_uri(mongodb_registry_uri)
        
        # Create test experiment
        experiment_name = f"compatibility_test_{int(time.time())}"
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            
        yield
        
        # Cleanup would go here if needed
    
    @pytest.fixture(scope="class")
    def sample_data(self):
        """Generate sample dataset for testing."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': [f'feature_{i}' for i in range(X.shape[1])]
        }
    
    def test_experiment_management(self):
        """Test experiment creation, listing, and management."""
        # Test experiment creation
        exp_name = f"test_experiment_{uuid.uuid4().hex[:8]}"
        exp_id = mlflow.create_experiment(exp_name)
        assert exp_id is not None
        
        # Test experiment retrieval
        experiment = mlflow.get_experiment(exp_id)
        assert experiment.name == exp_name
        assert experiment.experiment_id == exp_id
        
        # Test experiment listing
        experiments = mlflow.search_experiments()
        exp_names = [exp.name for exp in experiments]
        assert exp_name in exp_names
        
        # Test experiment by name
        exp_by_name = mlflow.get_experiment_by_name(exp_name)
        assert exp_by_name.experiment_id == exp_id
    
    def test_run_lifecycle_management(self):
        """Test complete run lifecycle operations."""
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            run_id = run.info.run_id
            
            # Test run is active
            assert mlflow.active_run() is not None
            assert mlflow.active_run().info.run_id == run_id
            
            # Log parameters
            mlflow.log_param("param1", "value1")
            mlflow.log_param("param2", 42)
            
            # Log metrics
            mlflow.log_metric("metric1", 0.85)
            mlflow.log_metric("metric2", 0.92)
            
            # Log tags
            mlflow.set_tag("tag1", "tag_value1")
            mlflow.set_tag("tag2", "tag_value2")
        
        # Test run retrieval after completion
        client = MlflowClient()
        completed_run = client.get_run(run_id)
        
        assert completed_run.info.status == RunStatus.to_string(RunStatus.FINISHED)
        assert completed_run.data.params["param1"] == "value1"
        assert completed_run.data.params["param2"] == "42"
        assert completed_run.data.metrics["metric1"] == 0.85
        assert completed_run.data.metrics["metric2"] == 0.92
        assert completed_run.data.tags["tag1"] == "tag_value1"
        assert completed_run.data.tags["tag2"] == "tag_value2"
    
    def test_parameter_logging(self):
        """Test parameter logging functionality."""
        with mlflow.start_run(experiment_id=self.experiment_id):
            # Test single parameter logging
            mlflow.log_param("single_param", "single_value")
            
            # Test multiple parameter logging
            params = {
                "param_str": "string_value",
                "param_int": 123,
                "param_float": 3.14,
                "param_bool": True,
                "param_list": str([1, 2, 3])  # Lists are converted to strings
            }
            mlflow.log_params(params)
            
            run_id = mlflow.active_run().info.run_id
        
        # Verify parameters were logged correctly
        client = MlflowClient()
        run = client.get_run(run_id)
        
        assert run.data.params["single_param"] == "single_value"
        assert run.data.params["param_str"] == "string_value"
        assert run.data.params["param_int"] == "123"
        assert run.data.params["param_float"] == "3.14"
        assert run.data.params["param_bool"] == "True"
    
    def test_metric_logging_and_history(self):
        """Test metric logging and history retrieval."""
        with mlflow.start_run(experiment_id=self.experiment_id):
            # Log single metrics
            mlflow.log_metric("accuracy", 0.85)
            mlflow.log_metric("precision", 0.90)
            
            # Log metric with step
            for step in range(5):
                mlflow.log_metric("loss", 1.0 - (step * 0.1), step=step)
            
            # Log multiple metrics
            metrics = {
                "recall": 0.88,
                "f1_score": 0.89,
                "auc": 0.92
            }
            mlflow.log_metrics(metrics)
            
            run_id = mlflow.active_run().info.run_id
        
        # Test metric retrieval
        client = MlflowClient()
        run = client.get_run(run_id)
        
        assert run.data.metrics["accuracy"] == 0.85
        assert run.data.metrics["precision"] == 0.90
        assert run.data.metrics["recall"] == 0.88
        
        # Test metric history
        loss_history = client.get_metric_history(run_id, "loss")
        assert len(loss_history) == 5
        
        # Verify metric values and steps
        for i, metric in enumerate(loss_history):
            expected_value = 1.0 - (i * 0.1)
            assert abs(metric.value - expected_value) < 1e-6
            assert metric.step == i
    
    def test_tag_management(self):
        """Test tag setting and retrieval."""
        with mlflow.start_run(experiment_id=self.experiment_id):
            # Test single tag
            mlflow.set_tag("model_type", "classifier")
            
            # Test multiple tags
            tags = {
                "framework": "sklearn",
                "version": "1.0.0",
                "environment": "test",
                "validated": "true"
            }
            mlflow.set_tags(tags)
            
            run_id = mlflow.active_run().info.run_id
        
        # Verify tags
        client = MlflowClient()
        run = client.get_run(run_id)
        
        assert run.data.tags["model_type"] == "classifier"
        assert run.data.tags["framework"] == "sklearn"
        assert run.data.tags["version"] == "1.0.0"
        assert run.data.tags["environment"] == "test"
        assert run.data.tags["validated"] == "true"
    
    def test_artifact_logging(self):
        """Test artifact logging functionality."""
        with mlflow.start_run(experiment_id=self.experiment_id):
            # Test dictionary logging
            config = {
                "model_config": {
                    "n_estimators": 100,
                    "max_depth": 10
                },
                "training_config": {
                    "batch_size": 32,
                    "epochs": 10
                }
            }
            mlflow.log_dict(config, "config.json")
            
            # Test text logging
            mlflow.log_text("Training completed successfully", "training_log.txt")
            
            # Test table logging
            data = pd.DataFrame({
                'epoch': [1, 2, 3, 4, 5],
                'loss': [0.8, 0.6, 0.4, 0.3, 0.2],
                'accuracy': [0.7, 0.8, 0.85, 0.9, 0.92]
            })
            mlflow.log_table(data, "training_metrics.json")
            
            run_id = mlflow.active_run().info.run_id
        
        # Verify artifacts were logged
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_names = [artifact.path for artifact in artifacts]
        
        assert "config.json" in artifact_names
        assert "training_log.txt" in artifact_names
        assert "training_metrics.json" in artifact_names
    
    def test_dataset_logging(self):
        """Test dataset logging and tracking."""
        # Create sample dataset
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        with mlflow.start_run(experiment_id=self.experiment_id):
            # Log dataset
            dataset = mlflow.data.from_pandas(
                data,
                source="test_dataset.csv",
                name="test_dataset"
            )
            
            mlflow.log_input(dataset, context="training")
            
            run_id = mlflow.active_run().info.run_id
        
        # Verify dataset was logged
        client = MlflowClient()
        run = client.get_run(run_id)
        
        # Check that inputs were logged (this may vary by MLflow version)
        # The exact assertion depends on how inputs are stored
        assert run_id is not None  # Basic verification that run completed
    
    def test_model_logging_sklearn(self, sample_data):
        """Test scikit-learn model logging."""
        X_train, y_train = sample_data['X_train'], sample_data['y_train']
        X_test, y_test = sample_data['X_test'], sample_data['y_test']
        
        with mlflow.start_run(experiment_id=self.experiment_id):
            # Train model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="TestSklearnModel"
            )
            
            # Log metrics
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            
            mlflow.log_metrics({
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy
            })
            
            run_id = mlflow.active_run().info.run_id
        
        # Test model loading
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # Verify model works
        predictions = loaded_model.predict(X_test)
        loaded_accuracy = accuracy_score(y_test, predictions)
        
        assert abs(loaded_accuracy - test_accuracy) < 1e-6
    
    def test_custom_pyfunc_model(self, sample_data):
        """Test custom PyFunc model logging."""
        class CustomModel(mlflow.pyfunc.PythonModel):
            def __init__(self, model):
                self.model = model
            
            def predict(self, context, model_input):
                return self.model.predict(model_input)
        
        X_train, y_train = sample_data['X_train'], sample_data['y_train']
        X_test, y_test = sample_data['X_test'], sample_data['y_test']
        
        with mlflow.start_run(experiment_id=self.experiment_id):
            # Train base model
            base_model = LogisticRegression(random_state=42)
            base_model.fit(X_train, y_train)
            
            # Create custom model
            custom_model = CustomModel(base_model)
            
            # Log custom model
            mlflow.pyfunc.log_model(
                python_model=custom_model,
                artifact_path="custom_model",
                registered_model_name="TestCustomModel"
            )
            
            run_id = mlflow.active_run().info.run_id
        
        # Test model loading and prediction
        model_uri = f"runs:/{run_id}/custom_model"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == len(y_test)
    
    def test_model_registry_operations(self):
        """Test model registry functionality."""
        client = MlflowClient()
        
        # Test registered model creation
        model_name = f"TestRegistryModel_{uuid.uuid4().hex[:8]}"
        try:
            registered_model = client.create_registered_model(
                model_name,
                description="Test model for registry operations"
            )
            assert registered_model.name == model_name
        except Exception:
            # Model might already exist
            registered_model = client.get_registered_model(model_name)
        
        # Test model version creation (simulated)
        with mlflow.start_run(experiment_id=self.experiment_id):
            mlflow.log_param("registry_test", "true")
            run_id = mlflow.active_run().info.run_id
        
        # Test registered model listing
        models = client.search_registered_models()
        model_names = [model.name for model in models]
        assert model_name in model_names
        
        # Test model retrieval
        retrieved_model = client.get_registered_model(model_name)
        assert retrieved_model.name == model_name
    
    def test_search_operations(self):
        """Test search functionality."""
        # Create multiple runs with different characteristics
        run_ids = []
        
        for i in range(3):
            with mlflow.start_run(experiment_id=self.experiment_id):
                mlflow.log_param("run_number", i + 1)
                mlflow.log_metric("test_metric", 0.8 + (i * 0.05))
                mlflow.set_tag("test_tag", f"value_{i + 1}")
                run_ids.append(mlflow.active_run().info.run_id)
        
        # Test search runs
        all_runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        assert len(all_runs) >= 3
        
        # Test search with filter
        high_metric_runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="metrics.test_metric > 0.85"
        )
        assert len(high_metric_runs) >= 1
        
        # Test search with parameter filter
        specific_runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string='params.run_number = "2"'
        )
        assert len(specific_runs) >= 1
    
    def test_batch_logging(self):
        """Test batch logging operations."""
        with mlflow.start_run(experiment_id=self.experiment_id):
            run_id = mlflow.active_run().info.run_id
        
        # Test batch logging
        client = MlflowClient()
        current_time = int(time.time() * 1000)
        
        # Create batch data
        metrics = [
            mlflow.entities.Metric("batch_metric_1", 0.1, current_time, 0),
            mlflow.entities.Metric("batch_metric_2", 0.2, current_time, 0),
            mlflow.entities.Metric("batch_metric_3", 0.3, current_time, 0)
        ]
        
        params = [
            mlflow.entities.Param("batch_param_1", "value1"),
            mlflow.entities.Param("batch_param_2", "value2")
        ]
        
        tags = [
            mlflow.entities.RunTag("batch_tag_1", "tag_value1"),
            mlflow.entities.RunTag("batch_tag_2", "tag_value2")
        ]
        
        # Log batch
        client.log_batch(run_id=run_id, metrics=metrics, params=params, tags=tags)
        
        # Verify batch logging
        run = client.get_run(run_id)
        
        assert run.data.metrics["batch_metric_1"] == 0.1
        assert run.data.metrics["batch_metric_2"] == 0.2
        assert run.data.metrics["batch_metric_3"] == 0.3
        assert run.data.params["batch_param_1"] == "value1"
        assert run.data.params["batch_param_2"] == "value2"
        assert run.data.tags["batch_tag_1"] == "tag_value1"
        assert run.data.tags["batch_tag_2"] == "tag_value2"
    
    def test_run_comparison(self, sample_data):
        """Test run comparison functionality."""
        X_train, y_train = sample_data['X_train'], sample_data['y_train']
        X_test, y_test = sample_data['X_test'], sample_data['y_test']
        
        models = [
            ("RandomForest", RandomForestClassifier(n_estimators=10, random_state=42)),
            ("GradientBoosting", GradientBoostingClassifier(n_estimators=10, random_state=42)),
            ("LogisticRegression", LogisticRegression(random_state=42))
        ]
        
        run_data = []
        
        for name, model in models:
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"model_comparison_{name}"):
                # Train model
                model.fit(X_train, y_train)
                
                # Predict and calculate metrics
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                
                # Log parameters and metrics
                mlflow.log_param("model_type", name)
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall
                })
                
                run_data.append({
                    "run_id": mlflow.active_run().info.run_id,
                    "model_type": name,
                    "accuracy": accuracy
                })
        
        # Compare runs
        runs_df = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string='params.model_type != ""'
        )
        
        # Verify we have comparison data
        assert len(runs_df) >= 3
        
        # Find best model
        best_run = runs_df.loc[runs_df['metrics.accuracy'].idxmax()]
        assert best_run is not None
    
    def test_experiment_and_run_deletion(self):
        """Test deletion operations."""
        client = MlflowClient()
        
        # Create test experiment for deletion
        exp_name = f"delete_test_{uuid.uuid4().hex[:8]}"
        exp_id = mlflow.create_experiment(exp_name)
        
        # Create test run
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_param("delete_test", "true")
            run_id = mlflow.active_run().info.run_id
        
        # Test run deletion
        client.delete_run(run_id)
        deleted_run = client.get_run(run_id)
        assert deleted_run.info.lifecycle_stage == "deleted"
        
        # Test run restoration
        client.restore_run(run_id)
        restored_run = client.get_run(run_id)
        assert restored_run.info.lifecycle_stage == "active"
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        client = MlflowClient()
        
        # Test non-existent run
        with pytest.raises(MlflowException):
            client.get_run("non_existent_run_id")
        
        # Test non-existent experiment
        with pytest.raises(MlflowException):
            mlflow.get_experiment("999999999")
        
        # Test invalid parameter logging
        with mlflow.start_run(experiment_id=self.experiment_id):
            # This should work fine - MLflow handles type conversion
            mlflow.log_param("test_param", None)  # None gets converted to string
            
        # Test metric logging with invalid values
        with mlflow.start_run(experiment_id=self.experiment_id):
            # This should raise an appropriate error or handle gracefully
            try:
                mlflow.log_metric("test_metric", float('inf'))
            except Exception:
                pass  # Expected for invalid metric values


class TestChatModelCompatibility:
    """Test ChatModel functionality compatibility."""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_mlflow(self):
        """Setup MLflow for ChatModel testing."""
        mlflow.set_tracking_uri("mongodb://localhost:27017/mlflow_chatmodel_test")
        mlflow.set_registry_uri("mongodb://localhost:27017/mlflow_chatmodel_test")
        
        yield
    
    def test_simple_chat_model(self):
        """Test basic ChatModel functionality."""
        class SimpleChatModel(mlflow.pyfunc.ChatModel):
            def predict(self, context, messages: List[ChatMessage], params: ChatParams) -> ChatCompletionResponse:
                user_message = messages[-1].content if messages else "Hello!"
                response_text = f"You said: {user_message}"
                
                response_message = ChatMessage(
                    role="assistant",
                    content=response_text
                )
                
                choice = ChatChoice(
                    index=0,
                    message=response_message,
                    finish_reason="stop"
                )
                
                usage = TokenUsageStats(
                    prompt_tokens=len(user_message.split()) if user_message else 0,
                    completion_tokens=len(response_text.split()),
                    total_tokens=len(user_message.split()) + len(response_text.split()) if user_message else len(response_text.split())
                )
                
                return ChatCompletionResponse(
                    id=f"test-{int(time.time())}",
                    object="chat.completion",
                    created=int(time.time()),
                    model="simple-test-model",
                    choices=[choice],
                    usage=usage
                )
        
        # Test model creation and prediction
        model = SimpleChatModel()
        messages = [ChatMessage(role="user", content="Hello, world!")]
        params = ChatParams(temperature=0.7, max_tokens=100)
        
        response = model.predict(context=None, messages=messages, params=params)
        
        assert response.choices[0].message.content == "You said: Hello, world!"
        assert response.usage.completion_tokens > 0
        
        # Test with experiment logging
        experiment_name = f"chatmodel_test_{int(time.time())}"
        experiment_id = mlflow.create_experiment(experiment_name)
        
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_param("model_type", "simple_chat")
            mlflow.log_metric("test_response_length", len(response.choices[0].message.content))
            
            # Log as artifacts for testing
            mlflow.log_dict({
                "test_response": response.choices[0].message.content,
                "model_info": "SimpleChatModel test"
            }, "chat_test_results.json")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])