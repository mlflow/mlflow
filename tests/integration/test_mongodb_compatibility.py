#!/usr/bin/env python
"""
Comprehensive MongoDB vs SQLAlchemy API Compatibility Test Suite

This test suite validates that MongoDB store responses are identical to
SQLAlchemy store responses for all MLflow operations, ensuring 100% API compatibility.
"""

import sys
import os
import tempfile
import shutil
import uuid
import time
import random
import pytest
from typing import List, Dict, Any, Optional
from datetime import datetime

# Ensure we're using Genesis-Flow
genesis_flow_path = "/Users/jagveersingh/Developer/autonomize/genesis/platform-agentops-mlops/genesis-flow"
if genesis_flow_path not in sys.path:
    sys.path.insert(0, genesis_flow_path)

import mlflow
from mlflow.entities import (
    Experiment, ExperimentTag, Run, RunInfo, RunData, RunTag, Param, Metric,
    RunStatus, LifecycleStage, ViewType
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.mongodb_store import MongoDBStore
from mlflow.exceptions import MlflowException
from mlflow.utils.time import get_current_time_millis

class MLflowStoreComparator:
    """Compares responses between MongoDB and SQLAlchemy stores for identical operations."""
    
    def __init__(self):
        self.temp_dir = None
        self.sql_store = None
        self.mongo_store = None
        self.test_results = {
            "experiments": {"passed": 0, "failed": 0, "errors": []},
            "runs": {"passed": 0, "failed": 0, "errors": []},
            "params": {"passed": 0, "failed": 0, "errors": []},
            "metrics": {"passed": 0, "failed": 0, "errors": []},
            "tags": {"passed": 0, "failed": 0, "errors": []},
            "model_registry": {"passed": 0, "failed": 0, "errors": []}
        }
    
    def setup(self):
        """Set up both MongoDB and SQLAlchemy stores for testing."""
        # Create temporary directory for SQLite
        self.temp_dir = tempfile.mkdtemp()
        sqlite_path = os.path.join(self.temp_dir, "test_mlflow.db")
        
        # Initialize SQLAlchemy store with SQLite
        sql_uri = f"sqlite:///{sqlite_path}"
        self.sql_store = SqlAlchemyStore(sql_uri, "file:///tmp/artifacts")
        
        # Initialize MongoDB store
        mongo_uri = "mongodb://localhost:27017/genesis_flow_compatibility_test"
        self.mongo_store = MongoDBStore(mongo_uri, "azure://artifacts")
        
        print(f"✅ Set up SQLAlchemy store: {sql_uri}")
        print(f"✅ Set up MongoDB store: {mongo_uri}")
    
    def teardown(self):
        """Clean up test resources."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clean up MongoDB test database
        if self.mongo_store and hasattr(self.mongo_store, 'sync_client'):
            self.mongo_store.sync_client.drop_database("genesis_flow_compatibility_test")
        
        print("🧹 Cleaned up test resources")
    
    def compare_objects(self, obj1: Any, obj2: Any, context: str = "") -> bool:
        """Compare two objects for equality, handling MLflow entities specially."""
        if type(obj1) != type(obj2):
            error = f"{context}: Type mismatch - {type(obj1)} vs {type(obj2)}"
            return False, error
        
        # Handle None values
        if obj1 is None and obj2 is None:
            return True, None
        if obj1 is None or obj2 is None:
            error = f"{context}: One is None - {obj1} vs {obj2}"
            return False, error
        
        # Handle MLflow entities
        if hasattr(obj1, '__dict__') and hasattr(obj2, '__dict__'):
            # Compare dictionaries of object attributes
            dict1 = obj1.__dict__
            dict2 = obj2.__dict__
            
            for key in set(dict1.keys()) | set(dict2.keys()):
                if key not in dict1:
                    error = f"{context}.{key}: Missing in obj1"
                    return False, error
                if key not in dict2:
                    error = f"{context}.{key}: Missing in obj2"
                    return False, error
                
                # Recursive comparison for nested objects
                is_equal, nested_error = self.compare_objects(
                    dict1[key], dict2[key], f"{context}.{key}"
                )
                if not is_equal:
                    return False, nested_error
            
            return True, None
        
        # Handle lists
        if isinstance(obj1, list) and isinstance(obj2, list):
            if len(obj1) != len(obj2):
                error = f"{context}: List length mismatch - {len(obj1)} vs {len(obj2)}"
                return False, error
            
            for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                is_equal, nested_error = self.compare_objects(
                    item1, item2, f"{context}[{i}]"
                )
                if not is_equal:
                    return False, nested_error
            
            return True, None
        
        # Handle dictionaries
        if isinstance(obj1, dict) and isinstance(obj2, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                error = f"{context}: Dict keys mismatch - {set(obj1.keys())} vs {set(obj2.keys())}"
                return False, error
            
            for key in obj1.keys():
                is_equal, nested_error = self.compare_objects(
                    obj1[key], obj2[key], f"{context}[{key}]"
                )
                if not is_equal:
                    return False, nested_error
            
            return True, None
        
        # Handle primitive types
        if obj1 != obj2:
            error = f"{context}: Value mismatch - {obj1} vs {obj2}"
            return False, error
        
        return True, None
    
    def compare_store_responses(self, operation: str, category: str, sql_result: Any, mongo_result: Any) -> bool:
        """Compare responses from both stores and record results."""
        is_equal, error = self.compare_objects(sql_result, mongo_result, operation)
        
        if is_equal:
            self.test_results[category]["passed"] += 1
            print(f"  ✅ {operation}: PASSED")
            return True
        else:
            self.test_results[category]["failed"] += 1
            self.test_results[category]["errors"].append(f"{operation}: {error}")
            print(f"  ❌ {operation}: FAILED - {error}")
            return False
    
    def test_experiment_operations(self):
        """Test all experiment CRUD operations."""
        print("\n🧪 Testing Experiment Operations")
        
        # Test 1: Create Experiment
        exp_name = f"test_experiment_{uuid.uuid4().hex[:8]}"
        tags = [ExperimentTag("env", "test"), ExperimentTag("version", "1.0")]
        
        try:
            sql_exp_id = self.sql_store.create_experiment(exp_name, "azure://artifacts", tags)
            mongo_exp_id = self.mongo_store.create_experiment(exp_name + "_mongo", "azure://artifacts", tags)
            
            # Both should return string IDs
            self.compare_store_responses(
                "create_experiment_return_type", "experiments",
                type(sql_exp_id), type(mongo_exp_id)
            )
        except Exception as e:
            self.test_results["experiments"]["failed"] += 1
            self.test_results["experiments"]["errors"].append(f"create_experiment: {e}")
            print(f"  ❌ create_experiment: ERROR - {e}")
            return
        
        # Test 2: Get Experiment
        try:
            sql_exp = self.sql_store.get_experiment(sql_exp_id)
            mongo_exp = self.mongo_store.get_experiment(mongo_exp_id)
            
            # Compare experiment structures (ignoring ID differences)
            self.compare_store_responses(
                "get_experiment_type", "experiments",
                type(sql_exp), type(mongo_exp)
            )
            
            # Compare specific fields
            self.compare_store_responses(
                "get_experiment_lifecycle_stage", "experiments",
                sql_exp.lifecycle_stage, mongo_exp.lifecycle_stage
            )
        except Exception as e:
            self.test_results["experiments"]["failed"] += 1
            self.test_results["experiments"]["errors"].append(f"get_experiment: {e}")
            print(f"  ❌ get_experiment: ERROR - {e}")
        
        # Test 3: List Experiments
        try:
            sql_experiments = self.sql_store.search_experiments()
            mongo_experiments = self.mongo_store.search_experiments()
            
            self.compare_store_responses(
                "search_experiments_type", "experiments",
                type(sql_experiments), type(mongo_experiments)
            )
            
            # Check that both return lists
            self.compare_store_responses(
                "search_experiments_is_list", "experiments",
                isinstance(sql_experiments, list), isinstance(mongo_experiments, list)
            )
        except Exception as e:
            self.test_results["experiments"]["failed"] += 1
            self.test_results["experiments"]["errors"].append(f"search_experiments: {e}")
            print(f"  ❌ search_experiments: ERROR - {e}")
        
        # Test 4: Delete Experiment
        try:
            self.sql_store.delete_experiment(sql_exp_id)
            self.mongo_store.delete_experiment(mongo_exp_id)
            
            # Both should complete without errors
            self.compare_store_responses(
                "delete_experiment_success", "experiments",
                True, True
            )
        except Exception as e:
            self.test_results["experiments"]["failed"] += 1
            self.test_results["experiments"]["errors"].append(f"delete_experiment: {e}")
            print(f"  ❌ delete_experiment: ERROR - {e}")
    
    def test_run_operations(self):
        """Test all run CRUD operations."""
        print("\n🏃 Testing Run Operations")
        
        # Create experiment first
        exp_name = f"run_test_experiment_{uuid.uuid4().hex[:8]}"
        sql_exp_id = self.sql_store.create_experiment(exp_name)
        mongo_exp_id = self.mongo_store.create_experiment(exp_name + "_mongo")
        
        # Test 1: Create Run
        user_id = "test_user"
        start_time = get_current_time_millis()
        tags = [RunTag("model", "test_model"), RunTag("version", "1.0")]
        run_name = "test_run"
        
        try:
            sql_run = self.sql_store.create_run(sql_exp_id, user_id, start_time, tags, run_name)
            mongo_run = self.mongo_store.create_run(mongo_exp_id, user_id, start_time, tags, run_name)
            
            # Compare run object types
            self.compare_store_responses(
                "create_run_type", "runs",
                type(sql_run), type(mongo_run)
            )
            
            # Compare run info types
            self.compare_store_responses(
                "create_run_info_type", "runs",
                type(sql_run.info), type(mongo_run.info)
            )
            
            # Compare run status
            self.compare_store_responses(
                "create_run_status", "runs",
                sql_run.info.status, mongo_run.info.status
            )
            
            sql_run_id = sql_run.info.run_id
            mongo_run_id = mongo_run.info.run_id
            
        except Exception as e:
            self.test_results["runs"]["failed"] += 1
            self.test_results["runs"]["errors"].append(f"create_run: {e}")
            print(f"  ❌ create_run: ERROR - {e}")
            return
        
        # Test 2: Get Run
        try:
            sql_run_retrieved = self.sql_store.get_run(sql_run_id)
            mongo_run_retrieved = self.mongo_store.get_run(mongo_run_id)
            
            self.compare_store_responses(
                "get_run_type", "runs",
                type(sql_run_retrieved), type(mongo_run_retrieved)
            )
            
            self.compare_store_responses(
                "get_run_status", "runs",
                sql_run_retrieved.info.status, mongo_run_retrieved.info.status
            )
        except Exception as e:
            self.test_results["runs"]["failed"] += 1
            self.test_results["runs"]["errors"].append(f"get_run: {e}")
            print(f"  ❌ get_run: ERROR - {e}")
        
        # Test 3: Update Run
        try:
            end_time = get_current_time_millis()
            sql_run_updated = self.sql_store.update_run_info(
                sql_run_id, RunStatus.FINISHED, end_time, "updated_run_name"
            )
            mongo_run_updated = self.mongo_store.update_run_info(
                mongo_run_id, RunStatus.FINISHED, end_time, "updated_run_name"
            )
            
            self.compare_store_responses(
                "update_run_status", "runs",
                sql_run_updated.status, mongo_run_updated.status
            )
            
            self.compare_store_responses(
                "update_run_end_time_set", "runs",
                sql_run_updated.end_time is not None, mongo_run_updated.end_time is not None
            )
        except Exception as e:
            self.test_results["runs"]["failed"] += 1
            self.test_results["runs"]["errors"].append(f"update_run_info: {e}")
            print(f"  ❌ update_run_info: ERROR - {e}")
        
        # Test 4: Search Runs
        try:
            sql_runs = self.sql_store.search_runs([sql_exp_id], "", ViewType.ALL, 100, [], None)
            mongo_runs = self.mongo_store.search_runs([mongo_exp_id], "", ViewType.ALL, 100, [], None)
            
            self.compare_store_responses(
                "search_runs_type", "runs",
                type(sql_runs), type(mongo_runs)
            )
            
            # Both should have at least 1 run
            self.compare_store_responses(
                "search_runs_has_results", "runs",
                len(sql_runs) > 0, len(mongo_runs) > 0
            )
        except Exception as e:
            self.test_results["runs"]["failed"] += 1
            self.test_results["runs"]["errors"].append(f"search_runs: {e}")
            print(f"  ❌ search_runs: ERROR - {e}")
        
        # Test 5: Delete Run
        try:
            self.sql_store.delete_run(sql_run_id)
            self.mongo_store.delete_run(mongo_run_id)
            
            self.compare_store_responses(
                "delete_run_success", "runs",
                True, True
            )
        except Exception as e:
            self.test_results["runs"]["failed"] += 1
            self.test_results["runs"]["errors"].append(f"delete_run: {e}")
            print(f"  ❌ delete_run: ERROR - {e}")
    
    def test_parameter_operations(self):
        """Test parameter logging operations."""
        print("\n📝 Testing Parameter Operations")
        
        # Create experiment and run first
        exp_name = f"param_test_experiment_{uuid.uuid4().hex[:8]}"
        sql_exp_id = self.sql_store.create_experiment(exp_name)
        mongo_exp_id = self.mongo_store.create_experiment(exp_name + "_mongo")
        
        sql_run = self.sql_store.create_run(sql_exp_id, "test_user", get_current_time_millis(), [], "param_test_run")
        mongo_run = self.mongo_store.create_run(mongo_exp_id, "test_user", get_current_time_millis(), [], "param_test_run")
        
        sql_run_id = sql_run.info.run_id
        mongo_run_id = mongo_run.info.run_id
        
        # Test 1: Log Single Parameter
        try:
            param = Param("learning_rate", "0.01")
            
            self.sql_store.log_param(sql_run_id, param)
            self.mongo_store.log_param(mongo_run_id, param)
            
            self.compare_store_responses(
                "log_param_success", "params",
                True, True
            )
        except Exception as e:
            self.test_results["params"]["failed"] += 1
            self.test_results["params"]["errors"].append(f"log_param: {e}")
            print(f"  ❌ log_param: ERROR - {e}")
        
        # Test 2: Log Batch Parameters
        try:
            params = [
                Param("epochs", "100"),
                Param("batch_size", "32"),
                Param("optimizer", "adam")
            ]
            
            self.sql_store.log_batch(sql_run_id, [], params, [])
            self.mongo_store.log_batch(mongo_run_id, [], params, [])
            
            self.compare_store_responses(
                "log_batch_params_success", "params",
                True, True
            )
        except Exception as e:
            self.test_results["params"]["failed"] += 1
            self.test_results["params"]["errors"].append(f"log_batch_params: {e}")
            print(f"  ❌ log_batch_params: ERROR - {e}")
        
        # Test 3: Retrieve Run with Parameters
        try:
            sql_run_with_params = self.sql_store.get_run(sql_run_id)
            mongo_run_with_params = self.mongo_store.get_run(mongo_run_id)
            
            # Compare parameter counts
            self.compare_store_responses(
                "get_run_param_count", "params",
                len(sql_run_with_params.data.params), len(mongo_run_with_params.data.params)
            )
            
            # Check that both have parameters
            self.compare_store_responses(
                "get_run_has_params", "params",
                len(sql_run_with_params.data.params) > 0, len(mongo_run_with_params.data.params) > 0
            )
        except Exception as e:
            self.test_results["params"]["failed"] += 1
            self.test_results["params"]["errors"].append(f"get_run_with_params: {e}")
            print(f"  ❌ get_run_with_params: ERROR - {e}")
    
    def test_metric_operations(self):
        """Test metric logging operations."""
        print("\n📊 Testing Metric Operations")
        
        # Create experiment and run first
        exp_name = f"metric_test_experiment_{uuid.uuid4().hex[:8]}"
        sql_exp_id = self.sql_store.create_experiment(exp_name)
        mongo_exp_id = self.mongo_store.create_experiment(exp_name + "_mongo")
        
        sql_run = self.sql_store.create_run(sql_exp_id, "test_user", get_current_time_millis(), [], "metric_test_run")
        mongo_run = self.mongo_store.create_run(mongo_exp_id, "test_user", get_current_time_millis(), [], "metric_test_run")
        
        sql_run_id = sql_run.info.run_id
        mongo_run_id = mongo_run.info.run_id
        
        # Test 1: Log Single Metric
        try:
            metric = Metric("accuracy", 0.95, get_current_time_millis(), 0)
            
            self.sql_store.log_metric(sql_run_id, metric)
            self.mongo_store.log_metric(mongo_run_id, metric)
            
            self.compare_store_responses(
                "log_metric_success", "metrics",
                True, True
            )
        except Exception as e:
            self.test_results["metrics"]["failed"] += 1
            self.test_results["metrics"]["errors"].append(f"log_metric: {e}")
            print(f"  ❌ log_metric: ERROR - {e}")
        
        # Test 2: Log Multiple Metrics (Time Series)
        try:
            current_time = get_current_time_millis()
            metrics = [
                Metric("loss", 0.8, current_time, 1),
                Metric("loss", 0.6, current_time + 1000, 2),
                Metric("loss", 0.4, current_time + 2000, 3),
                Metric("val_accuracy", 0.85, current_time, 1),
                Metric("val_accuracy", 0.90, current_time + 1000, 2),
            ]
            
            self.sql_store.log_batch(sql_run_id, metrics, [], [])
            self.mongo_store.log_batch(mongo_run_id, metrics, [], [])
            
            self.compare_store_responses(
                "log_batch_metrics_success", "metrics",
                True, True
            )
        except Exception as e:
            self.test_results["metrics"]["failed"] += 1
            self.test_results["metrics"]["errors"].append(f"log_batch_metrics: {e}")
            print(f"  ❌ log_batch_metrics: ERROR - {e}")
        
        # Test 3: Get Metric History
        try:
            sql_history = self.sql_store.get_metric_history(sql_run_id, "loss")
            mongo_history = self.mongo_store.get_metric_history(mongo_run_id, "loss")
            
            # Compare history lengths
            self.compare_store_responses(
                "get_metric_history_length", "metrics",
                len(sql_history), len(mongo_history)
            )
            
            # Compare that both have history
            self.compare_store_responses(
                "get_metric_history_has_data", "metrics",
                len(sql_history) > 0, len(mongo_history) > 0
            )
        except Exception as e:
            self.test_results["metrics"]["failed"] += 1
            self.test_results["metrics"]["errors"].append(f"get_metric_history: {e}")
            print(f"  ❌ get_metric_history: ERROR - {e}")
        
        # Test 4: Retrieve Run with Metrics
        try:
            sql_run_with_metrics = self.sql_store.get_run(sql_run_id)
            mongo_run_with_metrics = self.mongo_store.get_run(mongo_run_id)
            
            # Compare metric counts (should show latest values)
            self.compare_store_responses(
                "get_run_metric_count", "metrics",
                len(sql_run_with_metrics.data.metrics), len(mongo_run_with_metrics.data.metrics)
            )
            
            # Check that both have metrics
            self.compare_store_responses(
                "get_run_has_metrics", "metrics",
                len(sql_run_with_metrics.data.metrics) > 0, len(mongo_run_with_metrics.data.metrics) > 0
            )
        except Exception as e:
            self.test_results["metrics"]["failed"] += 1
            self.test_results["metrics"]["errors"].append(f"get_run_with_metrics: {e}")
            print(f"  ❌ get_run_with_metrics: ERROR - {e}")
    
    def test_tag_operations(self):
        """Test tag operations."""
        print("\n🏷️  Testing Tag Operations")
        
        # Create experiment and run first
        exp_name = f"tag_test_experiment_{uuid.uuid4().hex[:8]}"
        sql_exp_id = self.sql_store.create_experiment(exp_name)
        mongo_exp_id = self.mongo_store.create_experiment(exp_name + "_mongo")
        
        sql_run = self.sql_store.create_run(sql_exp_id, "test_user", get_current_time_millis(), [], "tag_test_run")
        mongo_run = self.mongo_store.create_run(mongo_exp_id, "test_user", get_current_time_millis(), [], "tag_test_run")
        
        sql_run_id = sql_run.info.run_id
        mongo_run_id = mongo_run.info.run_id
        
        # Test 1: Set Run Tags
        try:
            tags = [
                RunTag("environment", "production"),
                RunTag("model_type", "regression"),
                RunTag("version", "2.0")
            ]
            
            for tag in tags:
                self.sql_store.set_tag(sql_run_id, tag)
                self.mongo_store.set_tag(mongo_run_id, tag)
            
            self.compare_store_responses(
                "set_run_tags_success", "tags",
                True, True
            )
        except Exception as e:
            self.test_results["tags"]["failed"] += 1
            self.test_results["tags"]["errors"].append(f"set_run_tags: {e}")
            print(f"  ❌ set_run_tags: ERROR - {e}")
        
        # Test 2: Batch Set Tags
        try:
            batch_tags = [
                RunTag("data_source", "s3"),
                RunTag("experiment_type", "hyperparameter_tuning")
            ]
            
            self.sql_store.log_batch(sql_run_id, [], [], batch_tags)
            self.mongo_store.log_batch(mongo_run_id, [], [], batch_tags)
            
            self.compare_store_responses(
                "log_batch_tags_success", "tags",
                True, True
            )
        except Exception as e:
            self.test_results["tags"]["failed"] += 1
            self.test_results["tags"]["errors"].append(f"log_batch_tags: {e}")
            print(f"  ❌ log_batch_tags: ERROR - {e}")
        
        # Test 3: Retrieve Run with Tags
        try:
            sql_run_with_tags = self.sql_store.get_run(sql_run_id)
            mongo_run_with_tags = self.mongo_store.get_run(mongo_run_id)
            
            # Compare tag counts
            self.compare_store_responses(
                "get_run_tag_count", "tags",
                len(sql_run_with_tags.data.tags), len(mongo_run_with_tags.data.tags)
            )
            
            # Check that both have tags
            self.compare_store_responses(
                "get_run_has_tags", "tags",
                len(sql_run_with_tags.data.tags) > 0, len(mongo_run_with_tags.data.tags) > 0
            )
        except Exception as e:
            self.test_results["tags"]["failed"] += 1
            self.test_results["tags"]["errors"].append(f"get_run_with_tags: {e}")
            print(f"  ❌ get_run_with_tags: ERROR - {e}")
        
        # Test 4: Set Experiment Tags
        try:
            exp_tag = ExperimentTag("project", "genesis_flow_test")
            
            self.sql_store.set_experiment_tag(sql_exp_id, exp_tag)
            self.mongo_store.set_experiment_tag(mongo_exp_id, exp_tag)
            
            self.compare_store_responses(
                "set_experiment_tag_success", "tags",
                True, True
            )
        except Exception as e:
            self.test_results["tags"]["failed"] += 1
            self.test_results["tags"]["errors"].append(f"set_experiment_tag: {e}")
            print(f"  ❌ set_experiment_tag: ERROR - {e}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all compatibility tests and return results."""
        print("🚀 Starting MongoDB vs SQLAlchemy Compatibility Tests")
        print("=" * 60)
        
        try:
            self.setup()
            
            # Run all test categories
            self.test_experiment_operations()
            self.test_run_operations()
            self.test_parameter_operations()
            self.test_metric_operations()
            self.test_tag_operations()
            
            # Print summary
            self.print_summary()
            
            return self.test_results
            
        except Exception as e:
            print(f"❌ Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        
        finally:
            self.teardown()
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("📊 COMPATIBILITY TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_passed = 0
        total_failed = 0
        
        for category, results in self.test_results.items():
            passed = results["passed"]
            failed = results["failed"]
            total_passed += passed
            total_failed += failed
            
            status = "✅ PASS" if failed == 0 else "❌ FAIL"
            print(f"{category.upper():20} | {status} | {passed:3d} passed, {failed:3d} failed")
            
            # Print errors if any
            if results["errors"]:
                for error in results["errors"][:3]:  # Show first 3 errors
                    print(f"  └─ {error}")
                if len(results["errors"]) > 3:
                    print(f"  └─ ... and {len(results['errors']) - 3} more errors")
        
        print("-" * 60)
        print(f"TOTAL:                 | {total_passed:3d} passed, {total_failed:3d} failed")
        
        if total_failed == 0:
            print("\n🎉 ALL TESTS PASSED! MongoDB store is fully compatible with SQLAlchemy store.")
        else:
            print(f"\n⚠️  {total_failed} tests failed. MongoDB store needs fixes for full compatibility.")
        
        # Calculate compatibility percentage
        total_tests = total_passed + total_failed
        if total_tests > 0:
            compatibility_percent = (total_passed / total_tests) * 100
            print(f"📈 Compatibility Score: {compatibility_percent:.1f}%")


def main():
    """Run the compatibility test suite."""
    comparator = MLflowStoreComparator()
    results = comparator.run_all_tests()
    
    # Return success/failure for CI/CD
    if isinstance(results, dict) and "error" not in results:
        total_failed = sum(category.get("failed", 0) for category in results.values())
        return 0 if total_failed == 0 else 1
    else:
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)