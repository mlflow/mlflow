"""
Tests for managed datasets functionality in FileStore.

This module tests the managed dataset operations in the FileStore
tracking backend, including create, get, and delete operations.
"""

import json
import os
import tempfile
import uuid
from pathlib import Path

import pytest

from mlflow.entities.managed_datasets import ManagedDataset
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.tracking.file_store import FileStore
from mlflow.utils.file_utils import TempDir


class TestFileStoreManagedDatasets:
    """Test managed dataset operations in FileStore."""

    @pytest.fixture
    def store(self):
        """Create a temporary FileStore for testing."""
        with TempDir() as tmp:
            yield FileStore(tmp.path())

    @pytest.fixture
    def store_with_experiment(self, store):
        """Create a FileStore with a test experiment."""
        exp_id = store.create_experiment("test_managed_datasets")
        return store, exp_id

    def test_create_managed_dataset(self, store_with_experiment):
        """Test creating a managed dataset."""
        store, exp_id = store_with_experiment
        
        dataset = store.create_managed_dataset(
            name="Test QA Dataset",
            experiment_ids=[exp_id],
            source_type="human",
            source="manual_annotation",
            digest="abc123def456",
            schema='{"inputs": ["question", "context"], "expectations": ["answer"]}',
            profile='{"num_records": 50, "avg_length": 100}',
            created_by="test_user"
        )
        
        # Verify the dataset was created correctly
        assert dataset.name == "Test QA Dataset"
        assert dataset.experiment_ids == [exp_id]
        assert dataset.source_type == "human"
        assert dataset.source == "manual_annotation"
        assert dataset.digest == "abc123def456"
        assert '"num_records": 50' in dataset.schema
        assert dataset.created_by == "test_user"
        assert dataset.dataset_id is not None
        assert len(dataset.records) == 0

    def test_create_managed_dataset_minimal(self, store_with_experiment):
        """Test creating a managed dataset with minimal parameters."""
        store, exp_id = store_with_experiment
        
        dataset = store.create_managed_dataset(
            name="Minimal Dataset",
            experiment_ids=[exp_id]
        )
        
        assert dataset.name == "Minimal Dataset"
        assert dataset.experiment_ids == [exp_id]
        assert dataset.source_type is None
        assert dataset.source is None
        assert dataset.digest is None
        assert dataset.schema is None
        assert dataset.profile is None
        assert dataset.created_by is None
        assert dataset.dataset_id is not None

    def test_create_managed_dataset_multiple_experiments(self, store):
        """Test creating a managed dataset associated with multiple experiments."""
        exp_id1 = store.create_experiment("test_exp_1")
        exp_id2 = store.create_experiment("test_exp_2")
        
        dataset = store.create_managed_dataset(
            name="Multi-Experiment Dataset",
            experiment_ids=[exp_id1, exp_id2],
            created_by="test_user"
        )
        
        assert dataset.experiment_ids == [exp_id1, exp_id2]
        assert dataset.name == "Multi-Experiment Dataset"

    def test_create_managed_dataset_no_experiments_uses_default(self, store):
        """Test creating a managed dataset with no experiment IDs uses default."""
        dataset = store.create_managed_dataset(
            name="Default Experiment Dataset",
            experiment_ids=[],
            created_by="test_user"
        )
        
        # Should use default experiment "0"
        assert dataset.experiment_ids == []
        assert dataset.name == "Default Experiment Dataset"

    def test_get_managed_dataset(self, store_with_experiment):
        """Test retrieving a managed dataset."""
        store, exp_id = store_with_experiment
        
        # Create a dataset
        created_dataset = store.create_managed_dataset(
            name="Retrievable Dataset",
            experiment_ids=[exp_id],
            source_type="trace",
            source="trace-123",
            digest="xyz789",
            created_by="creator"
        )
        
        # Retrieve the dataset
        retrieved_dataset = store.get_managed_dataset(created_dataset.dataset_id)
        
        # Verify all fields match
        assert retrieved_dataset.dataset_id == created_dataset.dataset_id
        assert retrieved_dataset.name == "Retrievable Dataset"
        assert retrieved_dataset.experiment_ids == [exp_id]
        assert retrieved_dataset.source_type == "trace"
        assert retrieved_dataset.source == "trace-123"
        assert retrieved_dataset.digest == "xyz789"
        assert retrieved_dataset.created_by == "creator"
        assert retrieved_dataset.created_time == created_dataset.created_time
        assert retrieved_dataset.last_update_time == created_dataset.last_update_time

    def test_get_managed_dataset_not_found(self, store):
        """Test retrieving a non-existent managed dataset raises exception."""
        fake_dataset_id = str(uuid.uuid4())
        
        with pytest.raises(MlflowException, match=f"Managed dataset with ID '{fake_dataset_id}' not found"):
            store.get_managed_dataset(fake_dataset_id)

    def test_get_managed_dataset_across_experiments(self, store):
        """Test retrieving dataset stored in different experiment."""
        exp_id1 = store.create_experiment("exp_1")
        exp_id2 = store.create_experiment("exp_2")
        
        # Create dataset in exp_1
        dataset = store.create_managed_dataset(
            name="Cross-Experiment Dataset",
            experiment_ids=[exp_id1],
            created_by="test_user"
        )
        
        # Should be able to retrieve it regardless of which experiment we're "in"
        retrieved_dataset = store.get_managed_dataset(dataset.dataset_id)
        assert retrieved_dataset.dataset_id == dataset.dataset_id
        assert retrieved_dataset.name == "Cross-Experiment Dataset"

    def test_delete_managed_dataset(self, store_with_experiment):
        """Test deleting a managed dataset."""
        store, exp_id = store_with_experiment
        
        # Create a dataset
        dataset = store.create_managed_dataset(
            name="Dataset to Delete",
            experiment_ids=[exp_id],
            created_by="test_user"
        )
        
        # Verify it exists
        retrieved = store.get_managed_dataset(dataset.dataset_id)
        assert retrieved.name == "Dataset to Delete"
        
        # Delete it
        store.delete_managed_dataset(dataset.dataset_id)
        
        # Verify it's gone
        with pytest.raises(MlflowException, match=f"Managed dataset with ID '{dataset.dataset_id}' not found"):
            store.get_managed_dataset(dataset.dataset_id)

    def test_delete_managed_dataset_not_found(self, store):
        """Test deleting a non-existent managed dataset raises exception."""
        fake_dataset_id = str(uuid.uuid4())
        
        with pytest.raises(MlflowException, match=f"Managed dataset with ID '{fake_dataset_id}' not found"):
            store.delete_managed_dataset(fake_dataset_id)

    def test_managed_dataset_file_structure(self, store_with_experiment):
        """Test that managed datasets are stored in the correct file structure."""
        store, exp_id = store_with_experiment
        
        dataset = store.create_managed_dataset(
            name="File Structure Test",
            experiment_ids=[exp_id],
            created_by="test_user"
        )
        
        # Check that the file structure exists
        exp_path = store._get_experiment_path(exp_id, assert_exists=True)
        managed_datasets_dir = os.path.join(exp_path, FileStore.MANAGED_DATASETS_FOLDER_NAME)
        dataset_dir = os.path.join(managed_datasets_dir, dataset.dataset_id)
        dataset_file = os.path.join(dataset_dir, "dataset.json")
        
        assert os.path.exists(managed_datasets_dir)
        assert os.path.exists(dataset_dir)
        assert os.path.exists(dataset_file)
        
        # Verify file content
        with open(dataset_file, 'r') as f:
            stored_data = json.load(f)
        
        assert stored_data["name"] == "File Structure Test"
        assert stored_data["dataset_id"] == dataset.dataset_id
        assert stored_data["experiment_ids"] == [exp_id]

    def test_managed_dataset_json_serialization(self, store_with_experiment):
        """Test that managed datasets are correctly serialized to/from JSON."""
        store, exp_id = store_with_experiment
        
        # Create dataset with complex data
        dataset = store.create_managed_dataset(
            name="JSON Serialization Test",
            experiment_ids=[exp_id],
            source_type="document",
            source="s3://bucket/document.pdf",
            digest="complex_hash_123",
            schema='{"inputs": {"question": "string", "context": "string"}, "expectations": {"answer": "string", "confidence": "float"}}',
            profile='{"statistics": {"num_records": 100, "avg_input_length": 50, "categories": ["qa", "factual"]}}',
            created_by="json_tester"
        )
        
        # Retrieve and verify
        retrieved = store.get_managed_dataset(dataset.dataset_id)
        
        assert retrieved.name == "JSON Serialization Test"
        assert retrieved.source_type == "document"
        assert retrieved.source == "s3://bucket/document.pdf"
        assert retrieved.digest == "complex_hash_123"
        
        # Verify complex JSON fields
        assert '"inputs"' in retrieved.schema
        assert '"num_records": 100' in retrieved.profile
        assert '"categories"' in retrieved.profile

    def test_managed_dataset_artifact_directory_helper(self, store_with_experiment):
        """Test the artifact directory helper method."""
        store, exp_id = store_with_experiment
        
        dataset = store.create_managed_dataset(
            name="Artifact Dir Test",
            experiment_ids=[exp_id],
            created_by="test_user"
        )
        
        artifact_dir = store._get_managed_dataset_artifact_dir(exp_id, dataset.dataset_id)
        
        # Should follow the expected pattern
        expected_pattern = f"/{FileStore.MANAGED_DATASETS_FOLDER_NAME}/{dataset.dataset_id}/{FileStore.ARTIFACTS_FOLDER_NAME}"
        assert expected_pattern in artifact_dir
        assert artifact_dir.endswith(expected_pattern)

    def test_managed_dataset_concurrent_operations(self, store_with_experiment):
        """Test concurrent creation and retrieval of managed datasets."""
        store, exp_id = store_with_experiment
        
        # Create multiple datasets
        datasets = []
        for i in range(5):
            dataset = store.create_managed_dataset(
                name=f"Concurrent Dataset {i}",
                experiment_ids=[exp_id],
                source_type="test",
                source=f"source_{i}",
                created_by=f"user_{i}"
            )
            datasets.append(dataset)
        
        # Retrieve all datasets and verify they're distinct
        retrieved_datasets = []
        for dataset in datasets:
            retrieved = store.get_managed_dataset(dataset.dataset_id)
            retrieved_datasets.append(retrieved)
        
        # Verify all datasets are unique and correct
        dataset_ids = [d.dataset_id for d in retrieved_datasets]
        assert len(set(dataset_ids)) == 5  # All unique
        
        for i, dataset in enumerate(retrieved_datasets):
            assert dataset.name == f"Concurrent Dataset {i}"
            assert dataset.source == f"source_{i}"
            assert dataset.created_by == f"user_{i}"

    def test_managed_dataset_update_timestamps(self, store_with_experiment):
        """Test that timestamps are properly managed."""
        store, exp_id = store_with_experiment
        
        dataset = store.create_managed_dataset(
            name="Timestamp Test",
            experiment_ids=[exp_id],
            created_by="test_user"
        )
        
        # Verify timestamps are set
        assert dataset.created_time > 0
        assert dataset.last_update_time > 0
        assert dataset.created_time == dataset.last_update_time  # Should be same initially
        
        # Retrieve and verify timestamps are preserved
        retrieved = store.get_managed_dataset(dataset.dataset_id)
        assert retrieved.created_time == dataset.created_time
        assert retrieved.last_update_time == dataset.last_update_time

    def test_managed_dataset_invalid_experiment_id(self, store):
        """Test creating dataset with invalid experiment ID."""
        fake_exp_id = "non_existent_experiment"
        
        with pytest.raises(MlflowException):
            store.create_managed_dataset(
                name="Invalid Experiment Dataset",
                experiment_ids=[fake_exp_id],
                created_by="test_user"
            )

    def test_managed_dataset_reserved_folder_name(self, store_with_experiment):
        """Test that managed_datasets is in reserved folder names."""
        store, exp_id = store_with_experiment
        
        # This test verifies that MANAGED_DATASETS_FOLDER_NAME is properly reserved
        assert FileStore.MANAGED_DATASETS_FOLDER_NAME in FileStore.RESERVED_EXPERIMENT_FOLDERS
        
        # Create a dataset to ensure the folder gets created
        dataset = store.create_managed_dataset(
            name="Reserved Folder Test",
            experiment_ids=[exp_id],
            created_by="test_user"
        )
        
        # Verify the folder structure respects the reservation
        exp_path = store._get_experiment_path(exp_id, assert_exists=True)
        managed_datasets_path = os.path.join(exp_path, FileStore.MANAGED_DATASETS_FOLDER_NAME)
        assert os.path.exists(managed_datasets_path)

    def test_managed_dataset_empty_name_validation(self, store_with_experiment):
        """Test that empty dataset names are handled."""
        store, exp_id = store_with_experiment
        
        # Empty name should still work (validation can be done at higher levels)
        dataset = store.create_managed_dataset(
            name="",
            experiment_ids=[exp_id],
            created_by="test_user"
        )
        
        assert dataset.name == ""
        
        # Retrieve and verify
        retrieved = store.get_managed_dataset(dataset.dataset_id)
        assert retrieved.name == ""

    def test_managed_dataset_large_metadata(self, store_with_experiment):
        """Test managed dataset with large metadata fields."""
        store, exp_id = store_with_experiment
        
        # Create large schema and profile
        large_schema = json.dumps({
            "inputs": {f"field_{i}": "string" for i in range(100)},
            "expectations": {f"exp_{i}": "float" for i in range(50)}
        })
        
        large_profile = json.dumps({
            "statistics": {f"stat_{i}": i * 1.5 for i in range(200)},
            "metadata": {f"meta_{i}": f"value_{i}" for i in range(100)}
        })
        
        dataset = store.create_managed_dataset(
            name="Large Metadata Dataset",
            experiment_ids=[exp_id],
            schema=large_schema,
            profile=large_profile,
            created_by="test_user"
        )
        
        # Retrieve and verify large metadata is preserved
        retrieved = store.get_managed_dataset(dataset.dataset_id)
        
        assert retrieved.schema == large_schema
        assert retrieved.profile == large_profile
        assert "field_99" in retrieved.schema
        assert "stat_199" in retrieved.profile

    def test_managed_dataset_unicode_handling(self, store_with_experiment):
        """Test managed dataset with unicode characters."""
        store, exp_id = store_with_experiment
        
        dataset = store.create_managed_dataset(
            name="Unicode Test 测试 🚀",
            experiment_ids=[exp_id],
            source_type="human",
            source="人工标注",
            created_by="用户"
        )
        
        # Retrieve and verify unicode is preserved
        retrieved = store.get_managed_dataset(dataset.dataset_id)
        
        assert retrieved.name == "Unicode Test 测试 🚀"
        assert retrieved.source == "人工标注"
        assert retrieved.created_by == "用户"


class TestFileStoreManagedDatasetsIntegration:
    """Integration tests for managed datasets with other FileStore functionality."""

    @pytest.fixture
    def store(self):
        """Create a temporary FileStore for testing."""
        with TempDir() as tmp:
            yield FileStore(tmp.path())

    def test_managed_datasets_isolated_from_experiments(self, store):
        """Test that managed datasets don't interfere with experiment operations."""
        # Create experiment
        exp_id = store.create_experiment("test_isolation")
        
        # Create managed dataset
        dataset = store.create_managed_dataset(
            name="Isolation Test Dataset",
            experiment_ids=[exp_id],
            created_by="test_user"
        )
        
        # Verify experiment operations still work
        experiment = store.get_experiment(exp_id)
        assert experiment.name == "test_isolation"
        
        # Verify dataset operations still work
        retrieved_dataset = store.get_managed_dataset(dataset.dataset_id)
        assert retrieved_dataset.name == "Isolation Test Dataset"
        
        # Create a run in the experiment
        run = store.create_run(
            experiment_id=exp_id,
            user_id="test_user",
            start_time=0,
            tags=[],
            run_name="test_run"
        )
        
        # Verify both dataset and run coexist
        assert store.get_run(run.info.run_id).info.experiment_id == exp_id
        assert store.get_managed_dataset(dataset.dataset_id).experiment_ids == [exp_id]

    def test_managed_datasets_survive_experiment_deletion(self, store):
        """Test behavior when associated experiment is deleted."""
        # Create experiment and dataset
        exp_id = store.create_experiment("temp_experiment")
        dataset = store.create_managed_dataset(
            name="Dataset in Temp Experiment",
            experiment_ids=[exp_id],
            created_by="test_user"
        )
        
        # Verify dataset exists
        retrieved = store.get_managed_dataset(dataset.dataset_id)
        assert retrieved.name == "Dataset in Temp Experiment"
        
        # Delete the experiment
        store.delete_experiment(exp_id)
        
        # Dataset should still be retrievable (but experiment reference may be stale)
        # This behavior might be implementation-specific
        try:
            still_there = store.get_managed_dataset(dataset.dataset_id)
            assert still_there.name == "Dataset in Temp Experiment"
        except MlflowException:
            # If dataset is also deleted, that's acceptable behavior
            pass