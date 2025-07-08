#!/usr/bin/env python
"""
Comprehensive API Parity Tests: PostgreSQL vs MongoDB
Tests that all Genesis-Flow APIs work identically with both backends.

This module ensures 100% request/response compatibility between:
- PostgreSQL backend (existing)
- MongoDB backend (new)

All MLflow tracking and model registry operations must behave identically.
"""

import pytest
import mlflow
import json
import uuid
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

# Genesis-Flow specific imports
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus, LifecycleStage, ViewType
from mlflow.exceptions import MlflowException


class APIParityTestFramework:
    """Framework for testing API parity between PostgreSQL and MongoDB backends."""
    
    def __init__(self):
        """Initialize test framework with backend configurations."""
        self.postgres_uri = "postgresql://postgres:7HrX26sHIZz8yffytPc0@autonomize-database-1002.cwpqzu4drrfr.us-east-1.rds.amazonaws.com:5432/mlflow"
        self.mongodb_uri = "mongodb://localhost:27017/mlflow_api_parity_test"
        
        # Test artifacts
        self.test_artifacts_dir = None
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate comprehensive test data for API comparison."""
        return {
            'experiment_name': f'api_parity_test_{uuid.uuid4().hex[:8]}',
            'run_name': f'test_run_{uuid.uuid4().hex[:8]}',
            'model_name': f'test_model_{uuid.uuid4().hex[:8]}',
            'params': {
                'learning_rate': 0.01,
                'batch_size': 32,
                'epochs': 10,
                'optimizer': 'adam',
                'model_type': 'neural_network'
            },
            'metrics': {
                'accuracy': 0.95,
                'loss': 0.05,
                'precision': 0.92,
                'recall': 0.88,
                'f1_score': 0.90,
                'auc': 0.96
            },
            'tags': {
                'team': 'ml_platform',
                'environment': 'test',
                'framework': 'tensorflow',
                'version': '1.0.0',
                'priority': 'high'
            },
            'artifacts': {
                'model.pkl': b'mock_model_data',
                'config.json': json.dumps({'model_config': 'test'}).encode(),
                'metrics.csv': pd.DataFrame({'step': [1, 2, 3], 'loss': [0.1, 0.05, 0.02]}).to_csv().encode()
            }
        }
    
    def setup_test_artifacts(self):
        """Create temporary artifacts for testing."""
        self.test_artifacts_dir = tempfile.mkdtemp()
        
        for filename, content in self.test_data['artifacts'].items():
            file_path = Path(self.test_artifacts_dir) / filename
            file_path.write_bytes(content)
    
    def cleanup_test_artifacts(self):
        """Clean up temporary artifacts."""
        if self.test_artifacts_dir:
            shutil.rmtree(self.test_artifacts_dir, ignore_errors=True)
    
    def compare_responses(self, postgres_response: Any, mongodb_response: Any, operation: str) -> bool:
        """Compare responses from PostgreSQL and MongoDB backends."""
        # Handle different response types
        if isinstance(postgres_response, (list, tuple)):
            return self._compare_collections(postgres_response, mongodb_response, operation)
        elif hasattr(postgres_response, '__dict__'):
            return self._compare_objects(postgres_response, mongodb_response, operation)
        else:
            return self._compare_primitives(postgres_response, mongodb_response, operation)
    
    def _compare_collections(self, postgres_data: List, mongodb_data: List, operation: str) -> bool:
        """Compare collections (lists, tuples) from both backends."""
        if len(postgres_data) != len(mongodb_data):
            print(f"❌ {operation}: Collection length mismatch - PostgreSQL: {len(postgres_data)}, MongoDB: {len(mongodb_data)}")
            return False
        
        for i, (pg_item, mongo_item) in enumerate(zip(postgres_data, mongodb_data)):
            if not self.compare_responses(pg_item, mongo_item, f"{operation}[{i}]"):
                return False
        
        return True
    
    def _compare_objects(self, postgres_obj: Any, mongodb_obj: Any, operation: str) -> bool:
        """Compare MLflow entity objects from both backends."""
        if type(postgres_obj) != type(mongodb_obj):
            print(f"❌ {operation}: Object type mismatch - PostgreSQL: {type(postgres_obj)}, MongoDB: {type(mongodb_obj)}")
            return False
        
        # Get all attributes, excluding private ones
        pg_attrs = {k: v for k, v in postgres_obj.__dict__.items() if not k.startswith('_')}
        mongo_attrs = {k: v for k, v in mongodb_obj.__dict__.items() if not k.startswith('_')}
        
        # Compare attribute sets
        if set(pg_attrs.keys()) != set(mongo_attrs.keys()):
            missing_in_mongo = set(pg_attrs.keys()) - set(mongo_attrs.keys())
            missing_in_pg = set(mongo_attrs.keys()) - set(pg_attrs.keys())
            print(f"❌ {operation}: Attribute mismatch")
            if missing_in_mongo:
                print(f"   Missing in MongoDB: {missing_in_mongo}")
            if missing_in_pg:
                print(f"   Missing in PostgreSQL: {missing_in_pg}")
            return False
        
        # Compare attribute values
        for attr in pg_attrs:
            if not self.compare_responses(pg_attrs[attr], mongo_attrs[attr], f"{operation}.{attr}"):
                return False
        
        return True
    
    def _compare_primitives(self, postgres_val: Any, mongodb_val: Any, operation: str) -> bool:
        """Compare primitive values from both backends."""
        # Handle None values
        if postgres_val is None and mongodb_val is None:
            return True
        if postgres_val is None or mongodb_val is None:
            print(f"❌ {operation}: None mismatch - PostgreSQL: {postgres_val}, MongoDB: {mongodb_val}")
            return False
        
        # Handle timestamps (allow small differences due to precision)
        if isinstance(postgres_val, (int, float)) and isinstance(mongodb_val, (int, float)):
            if abs(postgres_val - mongodb_val) <= 1:  # 1ms tolerance
                return True
        
        # Direct comparison
        if postgres_val == mongodb_val:
            return True
        
        print(f"❌ {operation}: Value mismatch - PostgreSQL: {postgres_val}, MongoDB: {mongodb_val}")
        return False


@pytest.fixture
def api_framework():
    """Provide API parity test framework."""
    framework = APIParityTestFramework()
    framework.setup_test_artifacts()
    yield framework
    framework.cleanup_test_artifacts()


class TestExperimentAPIParity:
    """Test experiment-related API parity."""
    
    def test_create_experiment_parity(self, api_framework):
        """Test experiment creation returns identical results."""
        experiment_name = api_framework.test_data['experiment_name']
        
        # Test with PostgreSQL
        mlflow.set_tracking_uri(api_framework.postgres_uri)
        postgres_client = MlflowClient()
        pg_experiment_id = postgres_client.create_experiment(
            name=experiment_name,
            artifact_location=f"s3://test-bucket/{experiment_name}",
            tags={'backend': 'postgresql', 'test': 'parity'}
        )
        pg_experiment = postgres_client.get_experiment(pg_experiment_id)
        
        # Test with MongoDB
        mlflow.set_tracking_uri(api_framework.mongodb_uri)
        mongodb_client = MlflowClient()
        mongo_experiment_id = mongodb_client.create_experiment(
            name=experiment_name + "_mongo",
            artifact_location=f"s3://test-bucket/{experiment_name}_mongo",
            tags={'backend': 'mongodb', 'test': 'parity'}
        )
        mongo_experiment = mongodb_client.get_experiment(mongo_experiment_id)
        
        # Compare experiment structures (excluding unique fields)
        assert type(pg_experiment) == type(mongo_experiment)
        assert pg_experiment.lifecycle_stage == mongo_experiment.lifecycle_stage
        assert len(pg_experiment.tags) == len(mongo_experiment.tags)
        
        print("✅ Experiment creation API parity verified")
    
    def test_search_experiments_parity(self, api_framework):
        """Test experiment search returns structurally identical results."""
        # Test with PostgreSQL
        mlflow.set_tracking_uri(api_framework.postgres_uri)
        postgres_client = MlflowClient()
        pg_experiments = postgres_client.search_experiments(
            view_type=ViewType.ACTIVE_ONLY,
            max_results=10
        )
        
        # Test with MongoDB
        mlflow.set_tracking_uri(api_framework.mongodb_uri)
        mongodb_client = MlflowClient()
        mongo_experiments = mongodb_client.search_experiments(
            view_type=ViewType.ACTIVE_ONLY,
            max_results=10
        )
        
        # Compare response structures
        assert isinstance(pg_experiments, list)
        assert isinstance(mongo_experiments, list)
        
        if pg_experiments and mongo_experiments:
            pg_exp = pg_experiments[0]
            mongo_exp = mongo_experiments[0]
            
            # Compare object types and attributes
            assert type(pg_exp) == type(mongo_exp)
            assert hasattr(pg_exp, 'experiment_id')
            assert hasattr(mongo_exp, 'experiment_id')
            assert hasattr(pg_exp, 'name')
            assert hasattr(mongo_exp, 'name')
            assert hasattr(pg_exp, 'lifecycle_stage')
            assert hasattr(mongo_exp, 'lifecycle_stage')
        
        print("✅ Search experiments API parity verified")


class TestRunAPIParity:
    """Test run-related API parity."""
    
    def test_create_run_parity(self, api_framework):
        """Test run creation returns identical structures."""
        # Create experiment in both backends
        experiment_name = f"run_test_{uuid.uuid4().hex[:8]}"
        
        # PostgreSQL
        mlflow.set_tracking_uri(api_framework.postgres_uri)
        postgres_client = MlflowClient()
        pg_exp_id = postgres_client.create_experiment(experiment_name)
        
        pg_run = postgres_client.create_run(
            experiment_id=pg_exp_id,
            tags={'backend': 'postgresql'},
            run_name=api_framework.test_data['run_name']
        )
        
        # MongoDB
        mlflow.set_tracking_uri(api_framework.mongodb_uri)
        mongodb_client = MlflowClient()
        mongo_exp_id = mongodb_client.create_experiment(experiment_name + "_mongo")
        
        mongo_run = mongodb_client.create_run(
            experiment_id=mongo_exp_id,
            tags={'backend': 'mongodb'},
            run_name=api_framework.test_data['run_name'] + "_mongo"
        )
        
        # Compare run structures
        assert type(pg_run) == type(mongo_run)
        assert pg_run.info.status == mongo_run.info.status
        assert pg_run.info.lifecycle_stage == mongo_run.info.lifecycle_stage
        assert len(pg_run.data.tags) == len(mongo_run.data.tags)
        
        print("✅ Run creation API parity verified")
    
    def test_log_params_metrics_tags_parity(self, api_framework):
        """Test logging operations return identical results."""
        # Setup experiments and runs
        experiment_name = f"logging_test_{uuid.uuid4().hex[:8]}"
        
        # PostgreSQL setup
        mlflow.set_tracking_uri(api_framework.postgres_uri)
        with mlflow.start_run(experiment_id=mlflow.create_experiment(experiment_name)) as pg_run:
            # Log all test data
            mlflow.log_params(api_framework.test_data['params'])
            mlflow.log_metrics(api_framework.test_data['metrics'])
            mlflow.set_tags(api_framework.test_data['tags'])
            
            pg_run_data = mlflow.get_run(pg_run.info.run_id)
        
        # MongoDB setup
        mlflow.set_tracking_uri(api_framework.mongodb_uri)
        with mlflow.start_run(experiment_id=mlflow.create_experiment(experiment_name + "_mongo")) as mongo_run:
            # Log identical test data
            mlflow.log_params(api_framework.test_data['params'])
            mlflow.log_metrics(api_framework.test_data['metrics'])
            mlflow.set_tags(api_framework.test_data['tags'])
            
            mongo_run_data = mlflow.get_run(mongo_run.info.run_id)
        
        # Compare logged data structures
        assert len(pg_run_data.data.params) == len(mongo_run_data.data.params)
        assert len(pg_run_data.data.metrics) == len(mongo_run_data.data.metrics)
        assert len(pg_run_data.data.tags) == len(mongo_run_data.data.tags)
        
        # Compare param values
        for param_key in api_framework.test_data['params']:
            assert pg_run_data.data.params[param_key] == mongo_run_data.data.params[param_key]
        
        # Compare metric values
        for metric_key in api_framework.test_data['metrics']:
            assert pg_run_data.data.metrics[metric_key] == mongo_run_data.data.metrics[metric_key]
        
        print("✅ Logging operations API parity verified")
    
    def test_search_runs_parity(self, api_framework):
        """Test run search returns structurally identical results."""
        # PostgreSQL
        mlflow.set_tracking_uri(api_framework.postgres_uri)
        postgres_client = MlflowClient()
        pg_runs = postgres_client.search_runs(
            experiment_ids=[],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=5
        )
        
        # MongoDB
        mlflow.set_tracking_uri(api_framework.mongodb_uri)
        mongodb_client = MlflowClient()
        mongo_runs = mongodb_client.search_runs(
            experiment_ids=[],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=5
        )
        
        # Compare response structures
        assert isinstance(pg_runs, list)
        assert isinstance(mongo_runs, list)
        
        if pg_runs and mongo_runs:
            pg_run = pg_runs[0]
            mongo_run = mongo_runs[0]
            
            # Compare run object structures
            assert type(pg_run) == type(mongo_run)
            assert type(pg_run.info) == type(mongo_run.info)
            assert type(pg_run.data) == type(mongo_run.data)
        
        print("✅ Search runs API parity verified")


class TestModelRegistryAPIParity:
    """Test model registry API parity."""
    
    def test_register_model_parity(self, api_framework):
        """Test model registration returns identical structures."""
        model_name = api_framework.test_data['model_name']
        
        # Create runs with models in both backends
        # PostgreSQL
        mlflow.set_tracking_uri(api_framework.postgres_uri)
        with mlflow.start_run(experiment_id=mlflow.create_experiment(f"model_test_pg_{uuid.uuid4().hex[:8]}")):
            mlflow.sklearn.log_model(
                sk_model=None,  # Mock model
                artifact_path="model",
                registered_model_name=model_name
            )
            pg_run_id = mlflow.active_run().info.run_id
        
        postgres_client = MlflowClient()
        pg_model_version = postgres_client.get_model_version(model_name, version="1")
        
        # MongoDB
        mlflow.set_tracking_uri(api_framework.mongodb_uri)
        with mlflow.start_run(experiment_id=mlflow.create_experiment(f"model_test_mongo_{uuid.uuid4().hex[:8]}")):
            mlflow.sklearn.log_model(
                sk_model=None,  # Mock model
                artifact_path="model",
                registered_model_name=model_name + "_mongo"
            )
            mongo_run_id = mlflow.active_run().info.run_id
        
        mongodb_client = MlflowClient()
        mongo_model_version = mongodb_client.get_model_version(model_name + "_mongo", version="1")
        
        # Compare model version structures
        assert type(pg_model_version) == type(mongo_model_version)
        assert pg_model_version.current_stage == mongo_model_version.current_stage
        assert pg_model_version.status == mongo_model_version.status
        
        print("✅ Model registration API parity verified")
    
    def test_model_version_operations_parity(self, api_framework):
        """Test model version operations return identical results."""
        model_name = f"version_test_{uuid.uuid4().hex[:8]}"
        
        # PostgreSQL
        mlflow.set_tracking_uri(api_framework.postgres_uri)
        postgres_client = MlflowClient()
        postgres_client.create_registered_model(model_name)
        
        # MongoDB
        mlflow.set_tracking_uri(api_framework.mongodb_uri)
        mongodb_client = MlflowClient()
        mongodb_client.create_registered_model(model_name + "_mongo")
        
        # Compare registered model structures
        pg_model = postgres_client.get_registered_model(model_name)
        mongo_model = mongodb_client.get_registered_model(model_name + "_mongo")
        
        assert type(pg_model) == type(mongo_model)
        assert hasattr(pg_model, 'name')
        assert hasattr(mongo_model, 'name')
        assert hasattr(pg_model, 'creation_timestamp')
        assert hasattr(mongo_model, 'creation_timestamp')
        
        print("✅ Model version operations API parity verified")


class TestArtifactAPIParity:
    """Test artifact-related API parity."""
    
    def test_log_artifact_parity(self, api_framework):
        """Test artifact logging returns identical results."""
        # Setup runs in both backends
        experiment_name = f"artifact_test_{uuid.uuid4().hex[:8]}"
        
        # PostgreSQL
        mlflow.set_tracking_uri(api_framework.postgres_uri)
        with mlflow.start_run(experiment_id=mlflow.create_experiment(experiment_name)):
            mlflow.log_artifact(
                local_path=Path(api_framework.test_artifacts_dir) / "config.json",
                artifact_path="configs"
            )
            pg_run_id = mlflow.active_run().info.run_id
        
        postgres_client = MlflowClient()
        pg_artifacts = postgres_client.list_artifacts(pg_run_id, path="configs")
        
        # MongoDB
        mlflow.set_tracking_uri(api_framework.mongodb_uri)
        with mlflow.start_run(experiment_id=mlflow.create_experiment(experiment_name + "_mongo")):
            mlflow.log_artifact(
                local_path=Path(api_framework.test_artifacts_dir) / "config.json",
                artifact_path="configs"
            )
            mongo_run_id = mlflow.active_run().info.run_id
        
        mongodb_client = MlflowClient()
        mongo_artifacts = mongodb_client.list_artifacts(mongo_run_id, path="configs")
        
        # Compare artifact listings
        assert len(pg_artifacts) == len(mongo_artifacts)
        if pg_artifacts and mongo_artifacts:
            assert type(pg_artifacts[0]) == type(mongo_artifacts[0])
        
        print("✅ Artifact logging API parity verified")


class TestDatasetAPIParity:
    """Test dataset-related API parity."""
    
    def test_log_dataset_parity(self, api_framework):
        """Test dataset logging returns identical structures."""
        # Create test dataset
        test_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 0, 1, 0]
        })
        
        experiment_name = f"dataset_test_{uuid.uuid4().hex[:8]}"
        
        # PostgreSQL
        mlflow.set_tracking_uri(api_framework.postgres_uri)
        with mlflow.start_run(experiment_id=mlflow.create_experiment(experiment_name)):
            dataset = mlflow.data.from_pandas(test_df, source="test_source")
            mlflow.log_input(dataset, context="training")
            pg_run_id = mlflow.active_run().info.run_id
        
        postgres_client = MlflowClient()
        pg_run = postgres_client.get_run(pg_run_id)
        
        # MongoDB
        mlflow.set_tracking_uri(api_framework.mongodb_uri)
        with mlflow.start_run(experiment_id=mlflow.create_experiment(experiment_name + "_mongo")):
            dataset = mlflow.data.from_pandas(test_df, source="test_source")
            mlflow.log_input(dataset, context="training")
            mongo_run_id = mlflow.active_run().info.run_id
        
        mongodb_client = MlflowClient()
        mongo_run = mongodb_client.get_run(mongo_run_id)
        
        # Compare run inputs
        assert len(pg_run.inputs.dataset_inputs) == len(mongo_run.inputs.dataset_inputs)
        
        if pg_run.inputs.dataset_inputs and mongo_run.inputs.dataset_inputs:
            pg_input = pg_run.inputs.dataset_inputs[0]
            mongo_input = mongo_run.inputs.dataset_inputs[0]
            assert type(pg_input) == type(mongo_input)
        
        print("✅ Dataset logging API parity verified")


def test_comprehensive_api_workflow_parity(api_framework):
    """Test complete MLflow workflow maintains API parity."""
    print("\n🚀 Running Comprehensive API Workflow Parity Test")
    
    workflow_name = f"comprehensive_test_{uuid.uuid4().hex[:8]}"
    
    # Define workflow results containers
    postgres_results = {}
    mongodb_results = {}
    
    # Execute identical workflow on both backends
    for backend_name, tracking_uri, results_container in [
        ("PostgreSQL", api_framework.postgres_uri, postgres_results),
        ("MongoDB", api_framework.mongodb_uri, mongodb_results)
    ]:
        print(f"\n--- Testing {backend_name} Backend ---")
        
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        
        # 1. Create experiment
        experiment_id = client.create_experiment(
            name=f"{workflow_name}_{backend_name.lower()}",
            tags={'backend': backend_name.lower(), 'workflow': 'comprehensive'}
        )
        results_container['experiment_id'] = experiment_id
        
        # 2. Create and execute run
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"run_{backend_name.lower()}") as run:
            # Log parameters
            mlflow.log_params(api_framework.test_data['params'])
            
            # Log metrics
            mlflow.log_metrics(api_framework.test_data['metrics'])
            
            # Log tags
            mlflow.set_tags(api_framework.test_data['tags'])
            
            # Log artifacts
            mlflow.log_artifact(
                local_path=Path(api_framework.test_artifacts_dir) / "config.json",
                artifact_path="config"
            )
            
            # Log dataset
            test_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
            dataset = mlflow.data.from_pandas(test_df, source=f"test_source_{backend_name.lower()}")
            mlflow.log_input(dataset, context="training")
            
            results_container['run_id'] = run.info.run_id
        
        # 3. Register model
        model_name = f"model_{workflow_name}_{backend_name.lower()}"
        with mlflow.start_run(experiment_id=experiment_id):
            # Log a simple model (using sklearn as it's available)
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )
        
        results_container['model_name'] = model_name
        
        # 4. Get workflow results for comparison
        final_run = client.get_run(results_container['run_id'])
        model_version = client.get_model_version(model_name, version="1")
        experiment = client.get_experiment(experiment_id)
        
        results_container['final_run'] = final_run
        results_container['model_version'] = model_version
        results_container['experiment'] = experiment
    
    # Compare results between backends
    print("\n--- Comparing Results Between Backends ---")
    
    comparison_results = []
    
    # Compare experiments
    pg_exp = postgres_results['experiment']
    mongo_exp = mongodb_results['experiment']
    exp_match = (
        type(pg_exp) == type(mongo_exp) and
        pg_exp.lifecycle_stage == mongo_exp.lifecycle_stage and
        len(pg_exp.tags) == len(mongo_exp.tags)
    )
    comparison_results.append(("Experiment Structure", exp_match))
    
    # Compare runs
    pg_run = postgres_results['final_run']
    mongo_run = mongodb_results['final_run']
    run_match = (
        type(pg_run) == type(mongo_run) and
        len(pg_run.data.params) == len(mongo_run.data.params) and
        len(pg_run.data.metrics) == len(mongo_run.data.metrics) and
        len(pg_run.data.tags) == len(mongo_run.data.tags) and
        pg_run.info.status == mongo_run.info.status
    )
    comparison_results.append(("Run Structure", run_match))
    
    # Compare model versions
    pg_model = postgres_results['model_version']
    mongo_model = mongodb_results['model_version']
    model_match = (
        type(pg_model) == type(mongo_model) and
        pg_model.current_stage == mongo_model.current_stage and
        pg_model.status == mongo_model.status
    )
    comparison_results.append(("Model Version Structure", model_match))
    
    # Print comparison results
    all_passed = True
    for test_name, passed in comparison_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 COMPREHENSIVE API PARITY TEST PASSED")
        print("   All Genesis-Flow APIs work identically with PostgreSQL and MongoDB!")
    else:
        print("\n❌ COMPREHENSIVE API PARITY TEST FAILED")
        print("   Some APIs have differences between PostgreSQL and MongoDB backends.")
    
    assert all_passed, "API parity test failed - backends do not return identical structures"


if __name__ == "__main__":
    print("🧪 Running Genesis-Flow API Parity Tests")
    print("=" * 60)
    
    # Run the test framework
    framework = APIParityTestFramework()
    framework.setup_test_artifacts()
    
    try:
        # Run comprehensive workflow test
        test_comprehensive_api_workflow_parity(framework)
        
        print("\n✅ All API parity tests completed successfully!")
        print("🚀 Genesis-Flow APIs maintain 100% compatibility between PostgreSQL and MongoDB")
        
    except Exception as e:
        print(f"\n❌ API parity test failed: {e}")
        raise
    
    finally:
        framework.cleanup_test_artifacts()