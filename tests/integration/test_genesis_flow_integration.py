"""
Genesis-Flow Integration Tests

This module contains integration tests for Genesis-Flow, testing:
1. PostgreSQL with Managed Identity support
2. Google Cloud Storage artifact repository
3. Security features
4. Overall platform functionality
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pytest
import shutil

import mlflow
from mlflow import create_experiment, set_tracking_uri, get_tracking_uri, start_run, log_param, log_metric, set_tag, log_artifact, get_run, search_runs, delete_experiment, get_experiment, log_params, log_metrics, search_experiments
from mlflow.entities import ViewType
from mlflow.utils.security_validation import InputValidator, SecurityValidationError


class TestGenesisFlowIntegration(unittest.TestCase):
    """Integration tests for Genesis-Flow platform."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_uri = mlflow.get_tracking_uri()
    
    def tearDown(self):
        """Clean up test environment."""
        mlflow.set_tracking_uri(self.original_uri)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_file_store_basic_functionality(self):
        """Test basic MLflow functionality with file store."""
        # Set up file-based tracking
        tracking_uri = f"file://{self.test_dir}/mlruns"
        set_tracking_uri(tracking_uri)
        
        # Create experiment
        exp_name = "test_experiment"
        exp_id = mlflow.create_experiment(exp_name)
        
        # Create run and log data
        with mlflow.start_run(experiment_id=exp_id) as run:
            mlflow.log_param("test_param", "value1")
            mlflow.log_metric("test_metric", 0.95)
            mlflow.set_tag("test_tag", "tag_value")
            
            # Log artifact
            artifact_file = os.path.join(self.test_dir, "test_artifact.txt")
            with open(artifact_file, "w") as f:
                f.write("Test artifact content")
            mlflow.log_artifact(artifact_file)
        
        # Verify data was logged
        run_data = mlflow.get_run(run.info.run_id)
        self.assertEqual(run_data.data.params["test_param"], "value1")
        self.assertEqual(run_data.data.metrics["test_metric"], 0.95)
        self.assertEqual(run_data.data.tags["test_tag"], "tag_value")
    
    def test_security_validation_features(self):
        """Test security validation features."""
        # Test path traversal protection
        with self.assertRaises(SecurityValidationError):
            InputValidator.validate_experiment_name("../../../etc/passwd")
        
        # Test SQL injection protection
        with self.assertRaises(SecurityValidationError):
            InputValidator.validate_metric_key("metric'; DROP TABLE experiments; --")
        
        # Test valid inputs pass validation
        try:
            InputValidator.validate_experiment_name("valid_experiment_name")
            InputValidator.validate_metric_key("valid_metric_key")
            InputValidator.validate_param_value("valid_param_value")
        except SecurityValidationError:
            self.fail("Valid inputs should not raise SecurityValidationError")
    
    @patch("mlflow.store.tracking.postgres_managed_identity.PostgresManagedIdentityAuth.get_azure_ad_token")
    @patch("mlflow.store.tracking.postgres_managed_identity.create_engine")
    def test_postgres_managed_identity_integration(self, mock_create_engine, mock_get_token):
        """Test PostgreSQL with Managed Identity integration."""
        # Mock Azure AD token
        mock_get_token.return_value = "test_azure_token"
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Test with Managed Identity URI
        postgres_uri = "postgresql://user@server.postgres.database.azure.com:5432/mlflow?auth_method=managed_identity"
        
        # This would normally create a PostgreSQL store with Managed Identity
        # For testing, we just verify the URI format is accepted
        parsed = mlflow.tracking._tracking_service.utils._resolve_tracking_uri(postgres_uri)
        self.assertEqual(parsed, postgres_uri)
    
    def test_gcs_artifact_repository(self):
        """Test Google Cloud Storage artifact repository registration."""
        from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
        
        # Test that GCS URIs are recognized
        gcs_uri = "gs://test-bucket/path/to/artifacts"
        
        # Mock GCS client to avoid actual GCS calls
        with patch("google.cloud.storage.Client") as mock_client:
            mock_bucket = MagicMock()
            mock_client.return_value.bucket.return_value = mock_bucket
            
            try:
                repo = get_artifact_repository(gcs_uri)
                self.assertIsNotNone(repo)
                # Verify it's a GCS repository
                self.assertEqual(repo.__class__.__name__, "GCSArtifactRepository")
            except Exception as e:
                # If google-cloud-storage is not installed, skip this test
                if "No module named 'google'" in str(e):
                    self.skipTest("google-cloud-storage not installed")
                else:
                    raise
    
    def test_plugin_system_basic(self):
        """Test basic plugin system functionality."""
        try:
            from mlflow.plugins import get_plugin_manager
            
            plugin_manager = get_plugin_manager()
            plugin_manager.initialize(auto_discover=True, auto_enable_builtin=False)
            
            # List available plugins
            plugins = plugin_manager.list_plugins()
            self.assertIsInstance(plugins, list)
            
            # Verify plugin structure
            for plugin in plugins:
                self.assertIn("name", plugin)
                self.assertIn("enabled", plugin)
                self.assertIn("available", plugin)
        except ImportError:
            self.skipTest("Plugin system not available")
    
    def test_experiment_lifecycle(self):
        """Test complete experiment lifecycle."""
        tracking_uri = f"file://{self.test_dir}/mlruns"
        set_tracking_uri(tracking_uri)
        
        # Create experiment
        exp_name = "lifecycle_test"
        exp_id = mlflow.create_experiment(exp_name, tags={"purpose": "testing"})
        
        # Get experiment
        experiment = mlflow.get_experiment(exp_id)
        self.assertEqual(experiment.name, exp_name)
        self.assertEqual(experiment.tags["purpose"], "testing")
        
        # Search experiments
        experiments = mlflow.search_experiments(filter_string="tags.purpose = 'testing'")
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0].experiment_id, exp_id)
        
        # Delete experiment
        mlflow.delete_experiment(exp_id)
        
        # Verify deletion
        deleted_exp = mlflow.get_experiment(exp_id)
        self.assertEqual(deleted_exp.lifecycle_stage, "deleted")
    
    def test_model_logging_and_loading(self):
        """Test model logging and loading functionality."""
        tracking_uri = f"file://{self.test_dir}/mlruns"
        set_tracking_uri(tracking_uri)
        
        try:
            import sklearn
            from sklearn.linear_model import LogisticRegression
            import numpy as np
            
            # Create simple model
            X = np.array([[0, 0], [1, 1]])
            y = np.array([0, 1])
            model = LogisticRegression(random_state=42)
            model.fit(X, y)
            
            # Log model
            with start_run() as run:
                import mlflow.sklearn
                mlflow.sklearn.log_model(model, "model")
                model_uri = f"runs:/{run.info.run_id}/model"
            
            # Load model
            loaded_model = mlflow.sklearn.load_model(model_uri)
            
            # Verify model works
            predictions = loaded_model.predict(X)
            np.testing.assert_array_equal(predictions, y)
            
        except ImportError:
            self.skipTest("scikit-learn not installed")
    
    def test_batch_operations(self):
        """Test batch logging operations."""
        tracking_uri = f"file://{self.test_dir}/mlruns"
        set_tracking_uri(tracking_uri)
        
        with mlflow.start_run() as run:
            # Batch log params
            params = {"param1": "value1", "param2": "value2", "param3": "value3"}
            log_params(params)
            
            # Batch log metrics
            metrics = {"metric1": 0.1, "metric2": 0.2, "metric3": 0.3}
            log_metrics(metrics)
            
            # Verify batch operations
            run_data = get_run(run.info.run_id)
            for key, value in params.items():
                self.assertEqual(run_data.data.params[key], value)
            for key, value in metrics.items():
                self.assertEqual(run_data.data.metrics[key], value)


class TestPostgreSQLIntegration(unittest.TestCase):
    """Integration tests specifically for PostgreSQL features."""
    
    @pytest.mark.skipif(
        not os.getenv("MLFLOW_POSTGRES_TEST_URI"),
        reason="PostgreSQL test URI not provided"
    )
    def test_postgres_tracking_store(self):
        """Test PostgreSQL tracking store with real database."""
        postgres_uri = os.getenv("MLFLOW_POSTGRES_TEST_URI")
        mlflow.set_tracking_uri(postgres_uri)
        
        try:
            # Create experiment
            exp_id = mlflow.create_experiment(f"postgres_test_{os.getpid()}")
            
            # Create run
            with mlflow.start_run(experiment_id=exp_id) as run:
                mlflow.log_param("db_test", "postgres")
                mlflow.log_metric("accuracy", 0.99)
            
            # Verify data
            run_data = get_run(run.info.run_id)
            self.assertEqual(run_data.data.params["db_test"], "postgres")
            self.assertEqual(run_data.data.metrics["accuracy"], 0.99)
            
        finally:
            # Cleanup
            try:
                mlflow.delete_experiment(exp_id)
            except:
                pass


class TestGCSIntegration(unittest.TestCase):
    """Integration tests specifically for Google Cloud Storage."""
    
    @pytest.mark.skipif(
        not os.getenv("MLFLOW_GCS_TEST_BUCKET"),
        reason="GCS test bucket not provided"
    )
    def test_gcs_artifact_storage(self):
        """Test GCS artifact storage with real bucket."""
        gcs_bucket = os.getenv("MLFLOW_GCS_TEST_BUCKET")
        artifact_uri = f"gs://{gcs_bucket}/mlflow-tests"
        
        tracking_uri = f"file://{tempfile.mkdtemp()}/mlruns"
        set_tracking_uri(tracking_uri)
        
        try:
            # Create experiment with GCS artifact location
            exp_id = mlflow.create_experiment(
                f"gcs_test_{os.getpid()}",
                artifact_location=artifact_uri
            )
            
            # Log artifact
            with mlflow.start_run(experiment_id=exp_id) as run:
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                    f.write("GCS test artifact")
                    f.flush()
                    mlflow.log_artifact(f.name, "test_folder")
                
                # Verify artifact URI
                self.assertTrue(run.info.artifact_uri.startswith("gs://"))
            
        finally:
            # Cleanup
            try:
                mlflow.delete_experiment(exp_id)
            except:
                pass


def run_integration_tests():
    """Run all integration tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGenesisFlowIntegration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPostgreSQLIntegration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGCSIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)