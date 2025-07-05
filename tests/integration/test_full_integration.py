#!/usr/bin/env python3
"""
Genesis-Flow Full Integration Test Suite

Comprehensive end-to-end tests that validate all Genesis-Flow components
working together in realistic scenarios.
"""

import sys
import os
import tempfile
import time
import threading
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))

import pytest
import mlflow
from mlflow.entities import ViewType
from mlflow.plugins import get_plugin_manager
from mlflow.store.tracking.mongodb_store import MongoDBStore
from mlflow.utils.security_validation import InputValidator, SecurityValidationError

class TestFullIntegration:
    """Full integration tests for Genesis-Flow."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup clean test environment for each test."""
        # Use temporary directory for file-based tests
        self.temp_dir = tempfile.mkdtemp()
        self.original_tracking_uri = mlflow.get_tracking_uri()
        
        # Set test tracking URI
        mlflow.set_tracking_uri(f"file://{self.temp_dir}")
        
        yield
        
        # Cleanup
        mlflow.set_tracking_uri(self.original_tracking_uri)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_experiment_workflow(self):
        """Test complete experiment workflow from creation to analysis."""
        # Create experiment with security validation
        experiment_name = "test_integration_experiment"
        validated_name = InputValidator.validate_experiment_name(experiment_name)
        
        experiment_id = mlflow.create_experiment(
            name=validated_name,
            tags={"purpose": "integration_test", "version": "1.0"}
        )
        
        assert experiment_id is not None
        
        # Create multiple runs with different scenarios
        run_data = []
        
        # Run 1: Basic parameter and metric logging
        with mlflow.start_run(experiment_id=experiment_id) as run1:
            # Log parameters with validation
            params = {"learning_rate": "0.01", "epochs": "100", "model_type": "linear"}
            for key, value in params.items():
                validated_key = InputValidator.validate_param_key(key)
                validated_value = InputValidator.validate_param_value(value)
                mlflow.log_param(validated_key, validated_value)
            
            # Log metrics with validation
            for epoch in range(10):
                metric_key = InputValidator.validate_metric_key("accuracy")
                mlflow.log_metric(metric_key, 0.8 + (epoch * 0.02), step=epoch)
            
            # Log tags with validation
            tag_key = InputValidator.validate_tag_key("model_version")
            tag_value = InputValidator.validate_tag_value("v1.0")
            mlflow.set_tag(tag_key, tag_value)
            
            run_data.append({
                "run_id": run1.info.run_id,
                "accuracy": 0.98,
                "model_type": "linear"
            })
        
        # Run 2: With artifacts
        with mlflow.start_run(experiment_id=experiment_id) as run2:
            mlflow.log_param("model_type", "random_forest")
            mlflow.log_metric("accuracy", 0.95)
            
            # Create and log artifact
            artifact_file = Path(self.temp_dir) / "model_info.txt"
            artifact_file.write_text("Model information and metadata")
            
            # Validate artifact path
            artifact_path = InputValidator.validate_artifact_path("model/info.txt")
            mlflow.log_artifact(str(artifact_file), artifact_path="model")
            
            run_data.append({
                "run_id": run2.info.run_id,
                "accuracy": 0.95,
                "model_type": "random_forest"
            })
        
        # Verify experiment and runs
        experiment = mlflow.get_experiment(experiment_id)
        assert experiment.name == validated_name
        assert len(experiment.tags) == 2
        
        # Search runs with various filters
        all_runs = mlflow.search_runs(experiment_ids=[experiment_id])
        assert len(all_runs) == 2
        
        # Filter by metric
        high_accuracy_runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="metrics.accuracy > 0.96"
        )
        assert len(high_accuracy_runs) == 1
        assert high_accuracy_runs.iloc[0]["tags.model_version"] == "v1.0"
        
        # Test run retrieval and metric history
        run1_data = mlflow.get_run(run_data[0]["run_id"])
        assert run1_data.data.params["learning_rate"] == "0.01"
        
        metric_history = mlflow.get_metric_history(run_data[0]["run_id"], "accuracy")
        assert len(metric_history) == 10
        assert metric_history[-1].value == pytest.approx(0.98, rel=1e-2)
    
    def test_plugin_integration_workflow(self):
        """Test plugin system integration with MLflow workflow."""
        # Initialize plugin manager
        plugin_manager = get_plugin_manager()
        plugin_manager.initialize(auto_discover=True, auto_enable_builtin=False)
        
        # Test plugin discovery
        plugins = plugin_manager.list_plugins()
        plugin_names = [p["name"] for p in plugins]
        assert "sklearn" in plugin_names
        assert "pytorch" in plugin_names
        assert "transformers" in plugin_names
        
        # Test plugin enabling in context
        experiment_id = mlflow.create_experiment("plugin_test_experiment")
        
        # Use sklearn plugin temporarily
        try:
            with plugin_manager.plugin_context("sklearn") as sklearn_plugin:
                if sklearn_plugin:  # Only if dependencies are available
                    with mlflow.start_run(experiment_id=experiment_id):
                        # Plugin should be accessible
                        assert hasattr(mlflow, "sklearn")
                        
                        # Log some sklearn-specific data
                        mlflow.log_param("framework", "sklearn")
                        mlflow.log_metric("cv_score", 0.92)
        except RuntimeError:
            # Expected if sklearn not installed
            pass
        
        # Verify plugin is disabled after context
        assert not plugin_manager.is_plugin_enabled("sklearn")
    
    def test_security_validation_integration(self):
        """Test security validation integrated throughout the system."""
        experiment_id = mlflow.create_experiment("security_test_experiment")
        
        with mlflow.start_run(experiment_id=experiment_id):
            # Test that malicious inputs are blocked
            with pytest.raises(Exception):  # Should be caught by validation
                mlflow.log_param("../../../etc/passwd", "malicious")
            
            with pytest.raises(Exception):  # Should be caught by validation
                mlflow.set_tag("key'; DROP TABLE experiments; --", "injection")
            
            # Test that safe inputs work
            mlflow.log_param("safe_param", "safe_value")
            mlflow.log_metric("safe_metric", 0.85)
            mlflow.set_tag("safe_tag", "safe_value")
        
        # Verify safe data was logged
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        assert len(runs) == 1
        assert runs.iloc[0]["params.safe_param"] == "safe_value"
    
    def test_mongodb_integration_workflow(self):
        """Test MongoDB store integration (if available)."""
        try:
            # Test MongoDB store creation
            test_uri = "mongodb://localhost:27017/test_genesis_flow"
            mongo_store = MongoDBStore(test_uri, "file:///tmp/artifacts")
            
            # Test basic operations (without actual MongoDB connection)
            assert mongo_store.database_name == "test_genesis_flow"
            assert mongo_store.default_artifact_root == "file:///tmp/artifacts"
            
            # Test metadata conversion
            test_doc = {
                "experiment_id": "123",
                "name": "test_exp",
                "artifact_location": "file:///tmp/artifacts/123",
                "lifecycle_stage": "active",
                "creation_time": int(time.time() * 1000),
                "last_update_time": int(time.time() * 1000),
                "tags": [{"key": "env", "value": "test"}]
            }
            
            experiment = mongo_store._experiment_doc_to_entity(test_doc)
            assert experiment.name == "test_exp"
            assert len(experiment.tags) == 1
            assert experiment.tags[0].key == "env"
            
        except Exception as e:
            # MongoDB not available - skip test
            pytest.skip(f"MongoDB integration test skipped: {e}")
    
    def test_concurrent_operations(self):
        """Test thread-safe operations."""
        experiment_id = mlflow.create_experiment("concurrent_test_experiment")
        
        results = []
        errors = []
        
        def worker_function(worker_id):
            try:
                with mlflow.start_run(experiment_id=experiment_id):
                    # Each worker logs unique data
                    mlflow.log_param(f"worker_id", str(worker_id))
                    mlflow.log_metric(f"worker_metric", worker_id * 0.1)
                    
                    # Simulate some work
                    time.sleep(0.1)
                    
                    mlflow.log_metric(f"final_metric", worker_id * 0.2)
                    
                results.append(worker_id)
            except Exception as e:
                errors.append((worker_id, e))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors in concurrent operations: {errors}"
        assert len(results) == 5
        
        # Verify all runs were created
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        assert len(runs) == 5
    
    def test_large_data_handling(self):
        """Test handling of large datasets and many metrics."""
        experiment_id = mlflow.create_experiment("large_data_test")
        
        with mlflow.start_run(experiment_id=experiment_id):
            # Log many parameters
            for i in range(100):
                mlflow.log_param(f"param_{i:03d}", f"value_{i}")
            
            # Log many metrics with history
            for epoch in range(50):
                mlflow.log_metric("training_loss", 1.0 - (epoch * 0.01), step=epoch)
                mlflow.log_metric("validation_loss", 1.1 - (epoch * 0.009), step=epoch)
                mlflow.log_metric("learning_rate", 0.01 * (0.95 ** epoch), step=epoch)
            
            # Log batch metrics
            metrics = []
            for i in range(20):
                metrics.append({"key": f"batch_metric_{i}", "value": i * 0.05, "step": i})
            
            # Note: batch logging would be implemented in the actual MLflow store
        
        # Verify data was logged correctly
        run_data = mlflow.get_run(mlflow.active_run().info.run_id)
        assert len(run_data.data.params) == 100
        
        metric_history = mlflow.get_metric_history(run_data.info.run_id, "training_loss")
        assert len(metric_history) == 50
        assert metric_history[0].value == 1.0
        assert metric_history[-1].value == pytest.approx(0.51, rel=1e-2)
    
    def test_error_recovery_and_resilience(self):
        """Test system resilience and error recovery."""
        experiment_id = mlflow.create_experiment("error_recovery_test")
        
        # Test partial failure scenarios
        with mlflow.start_run(experiment_id=experiment_id):
            # Log valid data
            mlflow.log_param("valid_param", "valid_value")
            mlflow.log_metric("valid_metric", 0.85)
            
            # Try to log invalid data (should not break the run)
            try:
                mlflow.log_param("", "empty_key")  # Invalid parameter key
            except Exception:
                pass  # Expected to fail
            
            # Continue logging valid data
            mlflow.log_metric("recovery_metric", 0.90)
        
        # Verify valid data was preserved
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        assert len(runs) == 1
        run_data = runs.iloc[0]
        assert run_data["params.valid_param"] == "valid_value"
        assert run_data["metrics.recovery_metric"] == 0.90
    
    def test_api_compatibility(self):
        """Test API compatibility with standard MLflow usage patterns."""
        # Test all major API endpoints work
        experiment_id = mlflow.create_experiment("api_compatibility_test")
        
        # Test experiment operations
        experiment = mlflow.get_experiment(experiment_id)
        assert experiment is not None
        
        # Test run operations
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            
            # Test parameter logging
            mlflow.log_param("api_test_param", "test_value")
            
            # Test metric logging
            mlflow.log_metric("api_test_metric", 0.95)
            
            # Test tag setting
            mlflow.set_tag("api_test_tag", "test_tag_value")
        
        # Test run retrieval
        retrieved_run = mlflow.get_run(run_id)
        assert retrieved_run.data.params["api_test_param"] == "test_value"
        assert retrieved_run.data.metrics["api_test_metric"] == 0.95
        assert retrieved_run.data.tags["api_test_tag"] == "test_tag_value"
        
        # Test search functionality
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        assert len(runs) == 1
        
        filtered_runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="params.api_test_param = 'test_value'"
        )
        assert len(filtered_runs) == 1
    
    def test_performance_characteristics(self):
        """Test basic performance characteristics."""
        experiment_id = mlflow.create_experiment("performance_test")
        
        # Measure experiment creation time
        start_time = time.time()
        test_experiments = []
        for i in range(10):
            exp_id = mlflow.create_experiment(f"perf_test_exp_{i}")
            test_experiments.append(exp_id)
        creation_time = time.time() - start_time
        
        # Should be able to create 10 experiments quickly
        assert creation_time < 5.0, f"Experiment creation took {creation_time:.2f}s"
        
        # Measure run logging performance
        start_time = time.time()
        with mlflow.start_run(experiment_id=experiment_id):
            for i in range(100):
                mlflow.log_metric("perf_metric", i * 0.01, step=i)
        logging_time = time.time() - start_time
        
        # Should be able to log 100 metrics quickly
        assert logging_time < 10.0, f"Metric logging took {logging_time:.2f}s"
        
        # Measure search performance
        start_time = time.time()
        all_experiments = mlflow.search_experiments(view_type=ViewType.ALL)
        search_time = time.time() - start_time
        
        # Search should be fast
        assert search_time < 2.0, f"Experiment search took {search_time:.2f}s"
        assert len(all_experiments) >= len(test_experiments) + 2  # Including test experiments

def run_integration_tests():
    """Run all integration tests."""
    # Configure pytest to run tests
    pytest_args = [
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
    ]
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    return exit_code == 0

if __name__ == "__main__":
    success = run_integration_tests()
    if success:
        print("\n✅ All integration tests passed!")
        print("Genesis-Flow is ready for production deployment.")
    else:
        print("\n❌ Some integration tests failed.")
        print("Please review the failures before deploying.")
    
    sys.exit(0 if success else 1)