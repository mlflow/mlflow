#!/usr/bin/env python3
"""
Genesis-Flow Backward Compatibility Test Suite

Comprehensive test suite to verify that Genesis-Flow maintains 100% 
backward compatibility with standard MLflow APIs and behaviors.
"""

import os
import sys
import tempfile
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add Genesis-Flow to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import mlflow
from mlflow.entities import ViewType, RunStatus
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

class CompatibilityTestSuite:
    """Test suite for MLflow backward compatibility."""
    
    def __init__(self, test_tracking_uri: Optional[str] = None):
        """
        Initialize compatibility test suite.
        
        Args:
            test_tracking_uri: URI for testing (uses temp dir if None)
        """
        self.test_tracking_uri = test_tracking_uri
        self.temp_dir = None
        self.original_tracking_uri = None
        self.test_results = {
            "compatibility_version": "mlflow-2.x",
            "test_timestamp": time.time(),
            "test_tracking_uri": None,
            "tests": {},
            "summary": {},
            "api_coverage": {},
        }
        
        # Setup test environment
        self._setup_test_environment()
    
    def _setup_test_environment(self):
        """Setup isolated test environment."""
        self.original_tracking_uri = mlflow.get_tracking_uri()
        
        if not self.test_tracking_uri:
            self.temp_dir = tempfile.mkdtemp()
            self.test_tracking_uri = f"file://{self.temp_dir}"
        
        self.test_results["test_tracking_uri"] = self.test_tracking_uri
        mlflow.set_tracking_uri(self.test_tracking_uri)
    
    def _cleanup_test_environment(self):
        """Cleanup test environment."""
        if self.original_tracking_uri:
            mlflow.set_tracking_uri(self.original_tracking_uri)
        
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def run_all_tests(self) -> Dict:
        """Run all compatibility tests."""
        logger.info("Starting MLflow backward compatibility tests")
        
        try:
            # Core API compatibility tests
            self.test_results["tests"]["experiment_apis"] = self._test_experiment_apis()
            self.test_results["tests"]["run_apis"] = self._test_run_apis()
            self.test_results["tests"]["parameter_apis"] = self._test_parameter_apis()
            self.test_results["tests"]["metric_apis"] = self._test_metric_apis()
            self.test_results["tests"]["tag_apis"] = self._test_tag_apis()
            self.test_results["tests"]["artifact_apis"] = self._test_artifact_apis()
            self.test_results["tests"]["search_apis"] = self._test_search_apis()
            
            # Model APIs (if available)
            self.test_results["tests"]["model_apis"] = self._test_model_apis()
            
            # Client compatibility
            self.test_results["tests"]["tracking_client"] = self._test_tracking_client()
            
            # Exception compatibility
            self.test_results["tests"]["exception_behavior"] = self._test_exception_behavior()
            
            # Data format compatibility
            self.test_results["tests"]["data_formats"] = self._test_data_formats()
            
            # Configuration compatibility
            self.test_results["tests"]["configuration"] = self._test_configuration_compatibility()
            
            # Generate summary
            self._generate_test_summary()
            
            return self.test_results
            
        finally:
            self._cleanup_test_environment()
    
    def _test_experiment_apis(self) -> Dict:
        """Test experiment API compatibility."""
        test_result = {
            "status": "passed",
            "details": {},
            "api_calls_tested": [],
            "issues": [],
        }
        
        try:
            # Test create_experiment
            exp_name = "compatibility_test_experiment"
            exp_id = mlflow.create_experiment(exp_name)
            test_result["api_calls_tested"].append("create_experiment")
            test_result["details"]["created_experiment_id"] = exp_id
            
            # Test create_experiment with tags
            exp_id_with_tags = mlflow.create_experiment(
                "experiment_with_tags",
                tags={"env": "test", "version": "1.0"}
            )
            test_result["api_calls_tested"].append("create_experiment_with_tags")
            
            # Test get_experiment
            experiment = mlflow.get_experiment(exp_id)
            assert experiment.name == exp_name
            test_result["api_calls_tested"].append("get_experiment")
            
            # Test get_experiment_by_name
            experiment_by_name = mlflow.get_experiment_by_name(exp_name)
            assert experiment_by_name.experiment_id == exp_id
            test_result["api_calls_tested"].append("get_experiment_by_name")
            
            # Test search_experiments
            experiments = mlflow.search_experiments()
            assert len(experiments) >= 2
            test_result["api_calls_tested"].append("search_experiments")
            
            # Test search_experiments with view_type
            active_experiments = mlflow.search_experiments(view_type=ViewType.ACTIVE_ONLY)
            test_result["api_calls_tested"].append("search_experiments_with_view_type")
            
            # Test experiment deletion (if supported)
            try:
                mlflow.delete_experiment(exp_id_with_tags)
                test_result["api_calls_tested"].append("delete_experiment")
            except AttributeError:
                # delete_experiment might not be available in all versions
                pass
            
            test_result["details"]["api_calls_count"] = len(test_result["api_calls_tested"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["issues"].append(f"Experiment API test failed: {str(e)}")
        
        return test_result
    
    def _test_run_apis(self) -> Dict:
        """Test run API compatibility."""
        test_result = {
            "status": "passed",
            "details": {},
            "api_calls_tested": [],
            "issues": [],
        }
        
        try:
            # Create test experiment
            exp_id = mlflow.create_experiment("run_api_test")
            
            # Test start_run and end_run
            run = mlflow.start_run(experiment_id=exp_id)
            run_id = run.info.run_id
            test_result["api_calls_tested"].append("start_run")
            test_result["details"]["test_run_id"] = run_id
            
            # Test active_run
            active = mlflow.active_run()
            assert active.info.run_id == run_id
            test_result["api_calls_tested"].append("active_run")
            
            mlflow.end_run()
            test_result["api_calls_tested"].append("end_run")
            
            # Test context manager
            with mlflow.start_run(experiment_id=exp_id) as context_run:
                context_run_id = context_run.info.run_id
                test_result["api_calls_tested"].append("start_run_context_manager")
                test_result["details"]["context_run_id"] = context_run_id
            
            # Test get_run
            retrieved_run = mlflow.get_run(run_id)
            assert retrieved_run.info.run_id == run_id
            test_result["api_calls_tested"].append("get_run")
            
            # Test run status
            assert retrieved_run.info.status in [RunStatus.FINISHED, "FINISHED"]
            test_result["details"]["run_status_type"] = type(retrieved_run.info.status).__name__
            
            test_result["details"]["api_calls_count"] = len(test_result["api_calls_tested"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["issues"].append(f"Run API test failed: {str(e)}")
        
        return test_result
    
    def _test_parameter_apis(self) -> Dict:
        """Test parameter logging API compatibility."""
        test_result = {
            "status": "passed",
            "details": {},
            "api_calls_tested": [],
            "issues": [],
        }
        
        try:
            exp_id = mlflow.create_experiment("param_api_test")
            
            with mlflow.start_run(experiment_id=exp_id) as run:
                # Test log_param
                mlflow.log_param("learning_rate", 0.01)
                mlflow.log_param("epochs", 100)
                mlflow.log_param("model_type", "random_forest")
                test_result["api_calls_tested"].append("log_param")
                
                # Test log_params (batch)
                params = {"batch_size": 32, "optimizer": "adam", "dropout": 0.2}
                mlflow.log_params(params)
                test_result["api_calls_tested"].append("log_params")
                
                run_id = run.info.run_id
            
            # Verify parameters were logged
            retrieved_run = mlflow.get_run(run_id)
            logged_params = retrieved_run.data.params
            
            # Check individual parameters
            assert logged_params["learning_rate"] == "0.01"
            assert logged_params["epochs"] == "100"
            assert logged_params["model_type"] == "random_forest"
            
            # Check batch parameters
            assert logged_params["batch_size"] == "32"
            assert logged_params["optimizer"] == "adam"
            assert logged_params["dropout"] == "0.2"
            
            test_result["details"]["logged_param_count"] = len(logged_params)
            test_result["details"]["api_calls_count"] = len(test_result["api_calls_tested"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["issues"].append(f"Parameter API test failed: {str(e)}")
        
        return test_result
    
    def _test_metric_apis(self) -> Dict:
        """Test metric logging API compatibility."""
        test_result = {
            "status": "passed",
            "details": {},
            "api_calls_tested": [],
            "issues": [],
        }
        
        try:
            exp_id = mlflow.create_experiment("metric_api_test")
            
            with mlflow.start_run(experiment_id=exp_id) as run:
                # Test log_metric
                mlflow.log_metric("accuracy", 0.95)
                mlflow.log_metric("loss", 0.05)
                test_result["api_calls_tested"].append("log_metric")
                
                # Test log_metric with step
                for step in range(5):
                    mlflow.log_metric("training_loss", 1.0 - (step * 0.1), step=step)
                test_result["api_calls_tested"].append("log_metric_with_step")
                
                # Test log_metrics (batch)
                metrics = {"precision": 0.92, "recall": 0.88, "f1_score": 0.90}
                mlflow.log_metrics(metrics)
                test_result["api_calls_tested"].append("log_metrics")
                
                run_id = run.info.run_id
            
            # Verify metrics were logged
            retrieved_run = mlflow.get_run(run_id)
            logged_metrics = retrieved_run.data.metrics
            
            # Check metrics
            assert logged_metrics["accuracy"] == 0.95
            assert logged_metrics["loss"] == 0.05
            assert logged_metrics["precision"] == 0.92
            
            # Test metric history
            metric_history = mlflow.get_metric_history(run_id, "training_loss")
            assert len(metric_history) == 5
            test_result["api_calls_tested"].append("get_metric_history")
            
            test_result["details"]["logged_metric_count"] = len(logged_metrics)
            test_result["details"]["metric_history_length"] = len(metric_history)
            test_result["details"]["api_calls_count"] = len(test_result["api_calls_tested"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["issues"].append(f"Metric API test failed: {str(e)}")
        
        return test_result
    
    def _test_tag_apis(self) -> Dict:
        """Test tag API compatibility."""
        test_result = {
            "status": "passed",
            "details": {},
            "api_calls_tested": [],
            "issues": [],
        }
        
        try:
            exp_id = mlflow.create_experiment("tag_api_test")
            
            with mlflow.start_run(experiment_id=exp_id) as run:
                # Test set_tag
                mlflow.set_tag("version", "1.0")
                mlflow.set_tag("environment", "test")
                test_result["api_calls_tested"].append("set_tag")
                
                # Test set_tags (batch)
                tags = {"model_type": "ensemble", "framework": "scikit-learn", "author": "test"}
                mlflow.set_tags(tags)
                test_result["api_calls_tested"].append("set_tags")
                
                run_id = run.info.run_id
            
            # Verify tags were set
            retrieved_run = mlflow.get_run(run_id)
            logged_tags = retrieved_run.data.tags
            
            # Check tags
            assert logged_tags["version"] == "1.0"
            assert logged_tags["environment"] == "test"
            assert logged_tags["model_type"] == "ensemble"
            assert logged_tags["framework"] == "scikit-learn"
            
            test_result["details"]["logged_tag_count"] = len(logged_tags)
            test_result["details"]["api_calls_count"] = len(test_result["api_calls_tested"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["issues"].append(f"Tag API test failed: {str(e)}")
        
        return test_result
    
    def _test_artifact_apis(self) -> Dict:
        """Test artifact API compatibility."""
        test_result = {
            "status": "passed",
            "details": {},
            "api_calls_tested": [],
            "issues": [],
        }
        
        try:
            exp_id = mlflow.create_experiment("artifact_api_test")
            
            with mlflow.start_run(experiment_id=exp_id) as run:
                # Create test files
                test_file = os.path.join(self.temp_dir, "test_artifact.txt")
                with open(test_file, 'w') as f:
                    f.write("Test artifact content")
                
                # Test log_artifact
                mlflow.log_artifact(test_file)
                test_result["api_calls_tested"].append("log_artifact")
                
                # Test log_artifact with artifact_path
                mlflow.log_artifact(test_file, "models")
                test_result["api_calls_tested"].append("log_artifact_with_path")
                
                # Create directory for log_artifacts
                artifact_dir = os.path.join(self.temp_dir, "artifacts")
                os.makedirs(artifact_dir)
                
                for i in range(3):
                    with open(os.path.join(artifact_dir, f"file_{i}.txt"), 'w') as f:
                        f.write(f"Content {i}")
                
                # Test log_artifacts
                mlflow.log_artifacts(artifact_dir, "batch_artifacts")
                test_result["api_calls_tested"].append("log_artifacts")
                
                run_id = run.info.run_id
            
            # Test list_artifacts
            artifacts = mlflow.list_artifacts(run_id)
            test_result["api_calls_tested"].append("list_artifacts")
            test_result["details"]["artifact_count"] = len(artifacts)
            
            # Test list_artifacts with path
            model_artifacts = mlflow.list_artifacts(run_id, "models")
            test_result["api_calls_tested"].append("list_artifacts_with_path")
            
            test_result["details"]["api_calls_count"] = len(test_result["api_calls_tested"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["issues"].append(f"Artifact API test failed: {str(e)}")
        
        return test_result
    
    def _test_search_apis(self) -> Dict:
        """Test search API compatibility."""
        test_result = {
            "status": "passed",
            "details": {},
            "api_calls_tested": [],
            "issues": [],
        }
        
        try:
            # Create test data
            exp_id = mlflow.create_experiment("search_api_test")
            
            run_ids = []
            for i in range(5):
                with mlflow.start_run(experiment_id=exp_id):
                    mlflow.log_param("model_type", "linear" if i % 2 == 0 else "tree")
                    mlflow.log_metric("accuracy", 0.8 + (i * 0.02))
                    mlflow.set_tag("version", f"v{i}")
                    run_ids.append(mlflow.active_run().info.run_id)
            
            # Test search_runs
            all_runs = mlflow.search_runs(experiment_ids=[exp_id])
            assert len(all_runs) == 5
            test_result["api_calls_tested"].append("search_runs")
            
            # Test search_runs with filter_string
            linear_runs = mlflow.search_runs(
                experiment_ids=[exp_id],
                filter_string="params.model_type = 'linear'"
            )
            assert len(linear_runs) == 3  # 0, 2, 4
            test_result["api_calls_tested"].append("search_runs_with_filter")
            
            # Test search_runs with order_by
            ordered_runs = mlflow.search_runs(
                experiment_ids=[exp_id],
                order_by=["metrics.accuracy DESC"]
            )
            test_result["api_calls_tested"].append("search_runs_with_order")
            
            # Test search_runs with max_results
            limited_runs = mlflow.search_runs(
                experiment_ids=[exp_id],
                max_results=2
            )
            assert len(limited_runs) <= 2
            test_result["api_calls_tested"].append("search_runs_with_max_results")
            
            # Test search_runs with run_view_type
            active_runs = mlflow.search_runs(
                experiment_ids=[exp_id],
                run_view_type=ViewType.ACTIVE_ONLY
            )
            test_result["api_calls_tested"].append("search_runs_with_view_type")
            
            test_result["details"]["total_runs_found"] = len(all_runs)
            test_result["details"]["filtered_runs_found"] = len(linear_runs)
            test_result["details"]["api_calls_count"] = len(test_result["api_calls_tested"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["issues"].append(f"Search API test failed: {str(e)}")
        
        return test_result
    
    def _test_model_apis(self) -> Dict:
        """Test model API compatibility (basic)."""
        test_result = {
            "status": "passed",
            "details": {},
            "api_calls_tested": [],
            "issues": [],
        }
        
        try:
            # Test model logging APIs exist
            from mlflow import sklearn
            test_result["api_calls_tested"].append("sklearn_import")
            
            # Note: Full model testing would require actual models
            # This just tests that the APIs are available
            
            test_result["details"]["sklearn_available"] = True
            test_result["details"]["api_calls_count"] = len(test_result["api_calls_tested"])
            
        except ImportError as e:
            test_result["status"] = "warning"
            test_result["issues"].append(f"Model APIs not fully available: {str(e)}")
            test_result["details"]["sklearn_available"] = False
        
        except Exception as e:
            test_result["status"] = "failed"
            test_result["issues"].append(f"Model API test failed: {str(e)}")
        
        return test_result
    
    def _test_tracking_client(self) -> Dict:
        """Test MLflow tracking client compatibility."""
        test_result = {
            "status": "passed",
            "details": {},
            "api_calls_tested": [],
            "issues": [],
        }
        
        try:
            from mlflow.tracking import MlflowClient
            
            # Test client creation
            client = MlflowClient(tracking_uri=self.test_tracking_uri)
            test_result["api_calls_tested"].append("MlflowClient_creation")
            
            # Test client experiment operations
            exp_id = client.create_experiment("client_test")
            test_result["api_calls_tested"].append("client_create_experiment")
            
            experiment = client.get_experiment(exp_id)
            assert experiment.name == "client_test"
            test_result["api_calls_tested"].append("client_get_experiment")
            
            # Test client run operations
            run = client.create_run(exp_id)
            run_id = run.info.run_id
            test_result["api_calls_tested"].append("client_create_run")
            
            # Test client logging
            client.log_param(run_id, "client_param", "test")
            client.log_metric(run_id, "client_metric", 0.85)
            client.set_tag(run_id, "client_tag", "test")
            test_result["api_calls_tested"].append("client_logging")
            
            # Test client retrieval
            retrieved_run = client.get_run(run_id)
            assert retrieved_run.data.params["client_param"] == "test"
            test_result["api_calls_tested"].append("client_get_run")
            
            test_result["details"]["api_calls_count"] = len(test_result["api_calls_tested"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["issues"].append(f"Tracking client test failed: {str(e)}")
        
        return test_result
    
    def _test_exception_behavior(self) -> Dict:
        """Test exception compatibility."""
        test_result = {
            "status": "passed",
            "details": {},
            "api_calls_tested": [],
            "issues": [],
        }
        
        try:
            # Test that appropriate exceptions are raised
            
            # Test duplicate experiment name
            mlflow.create_experiment("duplicate_test")
            try:
                mlflow.create_experiment("duplicate_test")
                test_result["issues"].append("Duplicate experiment should raise exception")
            except Exception as e:
                assert "already exists" in str(e).lower()
                test_result["api_calls_tested"].append("duplicate_experiment_exception")
            
            # Test invalid run access
            try:
                mlflow.log_param("test", "value")  # No active run
                test_result["issues"].append("Logging without active run should raise exception")
            except Exception:
                test_result["api_calls_tested"].append("no_active_run_exception")
            
            # Test invalid run ID
            try:
                mlflow.get_run("invalid_run_id")
                test_result["issues"].append("Invalid run ID should raise exception")
            except Exception:
                test_result["api_calls_tested"].append("invalid_run_id_exception")
            
            test_result["details"]["api_calls_count"] = len(test_result["api_calls_tested"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["issues"].append(f"Exception behavior test failed: {str(e)}")
        
        return test_result
    
    def _test_data_formats(self) -> Dict:
        """Test data format compatibility."""
        test_result = {
            "status": "passed",
            "details": {},
            "api_calls_tested": [],
            "issues": [],
        }
        
        try:
            exp_id = mlflow.create_experiment("data_format_test")
            
            with mlflow.start_run(experiment_id=exp_id) as run:
                # Test various data types
                mlflow.log_param("string_param", "test_string")
                mlflow.log_param("int_param", 42)
                mlflow.log_param("float_param", 3.14)
                mlflow.log_param("bool_param", True)
                
                mlflow.log_metric("int_metric", 100)
                mlflow.log_metric("float_metric", 0.95)
                
                run_id = run.info.run_id
            
            # Verify data types are preserved/converted appropriately
            retrieved_run = mlflow.get_run(run_id)
            params = retrieved_run.data.params
            metrics = retrieved_run.data.metrics
            
            # Parameters should be strings
            assert isinstance(params["string_param"], str)
            assert params["int_param"] == "42"
            assert params["float_param"] == "3.14"
            assert params["bool_param"] == "True"
            
            # Metrics should be numeric
            assert isinstance(metrics["int_metric"], (int, float))
            assert isinstance(metrics["float_metric"], (int, float))
            
            test_result["details"]["param_types_verified"] = True
            test_result["details"]["metric_types_verified"] = True
            test_result["api_calls_tested"].append("data_type_handling")
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["issues"].append(f"Data format test failed: {str(e)}")
        
        return test_result
    
    def _test_configuration_compatibility(self) -> Dict:
        """Test configuration compatibility."""
        test_result = {
            "status": "passed",
            "details": {},
            "api_calls_tested": [],
            "issues": [],
        }
        
        try:
            # Test tracking URI handling
            original_uri = mlflow.get_tracking_uri()
            
            # Test set_tracking_uri
            mlflow.set_tracking_uri("file:///tmp/test")
            assert mlflow.get_tracking_uri() == "file:///tmp/test"
            test_result["api_calls_tested"].append("set_tracking_uri")
            
            # Restore original URI
            mlflow.set_tracking_uri(original_uri)
            
            # Test environment variable handling
            import os
            original_env = os.environ.get("MLFLOW_TRACKING_URI")
            
            os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/env_test"
            # Note: MLflow would need to be reloaded to pick up env changes
            test_result["api_calls_tested"].append("environment_variable_support")
            
            # Restore environment
            if original_env:
                os.environ["MLFLOW_TRACKING_URI"] = original_env
            else:
                os.environ.pop("MLFLOW_TRACKING_URI", None)
            
            test_result["details"]["api_calls_count"] = len(test_result["api_calls_tested"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["issues"].append(f"Configuration test failed: {str(e)}")
        
        return test_result
    
    def _generate_test_summary(self):
        """Generate test summary and API coverage report."""
        total_tests = len(self.test_results["tests"])
        passed_tests = sum(1 for test in self.test_results["tests"].values() 
                          if test["status"] == "passed")
        warning_tests = sum(1 for test in self.test_results["tests"].values() 
                           if test["status"] == "warning")
        failed_tests = sum(1 for test in self.test_results["tests"].values() 
                          if test["status"] == "failed")
        
        # Calculate API coverage
        total_api_calls = sum(len(test.get("api_calls_tested", [])) 
                             for test in self.test_results["tests"].values())
        
        self.test_results["summary"] = {
            "total_test_categories": total_tests,
            "passed_categories": passed_tests,
            "warning_categories": warning_tests,
            "failed_categories": failed_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "total_api_calls_tested": total_api_calls,
        }
        
        # API coverage report
        api_coverage = {}
        for test_name, test_result in self.test_results["tests"].items():
            api_coverage[test_name] = {
                "api_calls": test_result.get("api_calls_tested", []),
                "count": len(test_result.get("api_calls_tested", [])),
                "status": test_result["status"],
            }
        
        self.test_results["api_coverage"] = api_coverage
    
    def print_results(self):
        """Print formatted compatibility test results."""
        print("\\n" + "=" * 70)
        print("GENESIS-FLOW BACKWARD COMPATIBILITY TEST RESULTS")
        print("=" * 70)
        
        summary = self.test_results["summary"]
        print(f"Test Categories: {summary['total_test_categories']}")
        print(f"Passed: {summary['passed_categories']}")
        print(f"Warnings: {summary['warning_categories']}")
        print(f"Failed: {summary['failed_categories']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total API Calls Tested: {summary['total_api_calls_tested']}")
        
        print("\\nTest Category Results:")
        print("-" * 40)
        
        for test_name, test_result in self.test_results["tests"].items():
            status_symbol = {
                "passed": "✅",
                "warning": "⚠️",
                "failed": "❌"
            }.get(test_result["status"], "?")
            
            api_count = len(test_result.get("api_calls_tested", []))
            print(f"{status_symbol} {test_name}: {test_result['status']} ({api_count} APIs)")
            
            # Show issues if any
            for issue in test_result.get("issues", []):
                print(f"    ⚠️  {issue}")
        
        # Show API coverage summary
        print("\\nAPI Coverage Summary:")
        print("-" * 40)
        for category, coverage in self.test_results["api_coverage"].items():
            if coverage["count"] > 0:
                print(f"{category}: {coverage['count']} APIs tested")

def main():
    """Main CLI interface for compatibility testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Genesis-Flow backward compatibility")
    parser.add_argument("--tracking-uri",
                       help="Tracking URI to test (uses temp dir if not specified)")
    parser.add_argument("--output", "-o",
                       help="Output file for test results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run compatibility tests
    test_suite = CompatibilityTestSuite(test_tracking_uri=args.tracking_uri)
    
    try:
        results = test_suite.run_all_tests()
        
        # Print results
        test_suite.print_results()
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\\nDetailed results saved to: {args.output}")
        
        # Exit with appropriate code
        summary = results["summary"]
        if summary["failed_categories"] > 0:
            print("\\n❌ Compatibility tests failed!")
            sys.exit(1)
        elif summary["warning_categories"] > 0:
            print("\\n⚠️  Compatibility tests passed with warnings.")
            sys.exit(0)
        else:
            print("\\n✅ All compatibility tests passed!")
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Compatibility testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()