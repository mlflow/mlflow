import json
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd

from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.genai.datasets.databricks_evaluation_dataset_source import (
    DatabricksEvaluationDatasetSource,
)
from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset


def create_mock_managed_dataset(source_value: Any = "test-digest") -> Mock:
    """Create a mock Databricks Agent Evaluation ManagedDataset for testing"""
    mock_dataset = Mock()
    mock_dataset.dataset_id = "test-dataset-id"
    mock_dataset.name = "catalog.schema.table"
    mock_dataset.digest = "test-digest"
    mock_dataset.schema = "test-schema"
    mock_dataset.profile = "test-profile"
    mock_dataset.source = source_value
    mock_dataset.source_type = "databricks-uc-table"
    mock_dataset.create_time = "2024-01-01T00:00:00"
    mock_dataset.created_by = "test-user"
    mock_dataset.last_update_time = "2024-01-02T00:00:00"
    mock_dataset.last_updated_by = "test-user-2"

    # Mock methods
    mock_dataset.to_df.return_value = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    mock_dataset.set_profile.return_value = mock_dataset
    mock_dataset.merge_records.return_value = mock_dataset

    return mock_dataset


def create_dataset_with_source(source_value: Any) -> EvaluationDataset:
    """Factory function to create EvaluationDataset with specific source value."""
    mock_dataset = create_mock_managed_dataset(source_value)
    return EvaluationDataset(mock_dataset)


def test_evaluation_dataset_init():
    mock_managed_dataset = create_mock_managed_dataset()
    dataset = EvaluationDataset(mock_managed_dataset)
    assert dataset._dataset is mock_managed_dataset
    assert dataset._df is None
    assert dataset._digest is None


def test_evaluation_dataset_properties():
    dataset = create_dataset_with_source("test-digest")

    assert dataset.dataset_id == "test-dataset-id"
    assert dataset.name == "catalog.schema.table"
    assert dataset.digest == "test-digest"
    assert dataset.schema == "test-schema"
    assert dataset.profile == "test-profile"
    assert dataset.source_type == "databricks-uc-table"
    assert dataset.create_time == "2024-01-01T00:00:00"
    assert dataset.created_by == "test-user"
    assert dataset.last_update_time == "2024-01-02T00:00:00"
    assert dataset.last_updated_by == "test-user-2"


def test_evaluation_dataset_source_default():
    # Test that source returns DatabricksEvaluationDatasetSource with table name and dataset ID
    # when the managed dataset source is not a DatasetSource
    dataset = create_dataset_with_source("string-value")

    source = dataset.source
    assert isinstance(source, DatabricksEvaluationDatasetSource)
    assert source.table_name == "catalog.schema.table"  # Uses dataset.name
    assert source.dataset_id == "test-dataset-id"


def test_evaluation_dataset_source_with_none():
    # Test that source returns DatabricksEvaluationDatasetSource when managed dataset source is None
    dataset = create_dataset_with_source(None)

    source = dataset.source
    assert isinstance(source, DatabricksEvaluationDatasetSource)
    assert source.table_name == "catalog.schema.table"  # Uses dataset.name
    assert source.dataset_id == "test-dataset-id"


def test_evaluation_dataset_source_with_existing_dataset_source():
    # Test when source is already a DatasetSource object - should return it directly
    existing_source = DatabricksEvaluationDatasetSource(table_name="existing.table")
    dataset = create_dataset_with_source(existing_source)

    source = dataset.source
    assert source is existing_source


def test_evaluation_dataset_source_with_spark_dataset_source():
    # Test when source is a SparkDatasetSource - should return it directly
    spark_source = SparkDatasetSource(table_name="spark.table")
    dataset = create_dataset_with_source(spark_source)

    source = dataset.source
    assert source is spark_source


def test_evaluation_dataset_to_df():
    mock_managed_dataset = create_mock_managed_dataset()
    dataset = EvaluationDataset(mock_managed_dataset)

    # First call should fetch from managed dataset
    df = dataset.to_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    mock_managed_dataset.to_df.assert_called_once()

    # Second call should use cached version
    df2 = dataset.to_df()
    assert df2 is df
    assert mock_managed_dataset.to_df.call_count == 1  # Still only called once


def test_evaluation_dataset_to_mlflow_entity():
    # Test that _to_mlflow_entity properly serializes the source to JSON
    dataset = create_dataset_with_source("any-value")

    entity = dataset._to_mlflow_entity()
    assert entity.name == "catalog.schema.table"
    assert entity.digest == "test-digest"
    assert entity.source_type == "databricks-uc-table"

    # Check that source is properly serialized to JSON
    source_dict = json.loads(entity.source)
    assert source_dict["table_name"] == "catalog.schema.table"
    assert source_dict["dataset_id"] == "test-dataset-id"
    assert entity.schema == "test-schema"
    assert entity.profile == "test-profile"


def test_evaluation_dataset_to_mlflow_entity_with_existing_source():
    # Test that _to_mlflow_entity works with an existing DatasetSource
    existing_source = DatabricksEvaluationDatasetSource(
        table_name="existing.table", dataset_id="existing-id"
    )
    dataset = create_dataset_with_source(existing_source)

    entity = dataset._to_mlflow_entity()
    assert entity.name == "catalog.schema.table"
    assert entity.digest == "test-digest"
    assert entity.source_type == "databricks-uc-table"

    # Check that the existing source is properly serialized to JSON
    source_dict = json.loads(entity.source)
    assert source_dict["table_name"] == "existing.table"
    assert source_dict["dataset_id"] == "existing-id"
    assert entity.schema == "test-schema"
    assert entity.profile == "test-profile"


def test_evaluation_dataset_set_profile():
    mock_managed_dataset = create_mock_managed_dataset()
    dataset = EvaluationDataset(mock_managed_dataset)

    new_dataset = dataset.set_profile("new-profile")
    assert isinstance(new_dataset, EvaluationDataset)
    mock_managed_dataset.set_profile.assert_called_once_with("new-profile")


def test_evaluation_dataset_merge_records():
    mock_managed_dataset = create_mock_managed_dataset()
    dataset = EvaluationDataset(mock_managed_dataset)

    new_records = [{"col1": 4, "col2": "d"}]
    new_dataset = dataset.merge_records(new_records)
    assert isinstance(new_dataset, EvaluationDataset)
    mock_managed_dataset.merge_records.assert_called_once_with(new_records)


@patch("mlflow.genai.datasets.evaluation_dataset.compute_pandas_digest")
def test_evaluation_dataset_digest_computation(mock_compute_digest):
    # Test when managed dataset has no digest
    mock_managed_dataset = create_mock_managed_dataset()
    mock_managed_dataset.digest = None
    mock_compute_digest.return_value = "computed-digest"

    dataset = EvaluationDataset(mock_managed_dataset)
    digest = dataset.digest

    assert digest == "computed-digest"
    mock_compute_digest.assert_called_once()

    # Subsequent calls should use cached digest
    digest2 = dataset.digest
    assert digest2 == "computed-digest"
    assert mock_compute_digest.call_count == 1


def test_evaluation_dataset_to_evaluation_dataset():
    dataset = create_dataset_with_source("test-digest")

    legacy_dataset = dataset.to_evaluation_dataset(
        path="/path/to/data", feature_names=["col1", "col2"]
    )

    # Legacy EvaluationDataset stores data as _features_data
    assert legacy_dataset._features_data.equals(dataset.to_df())
    assert legacy_dataset._path == "/path/to/data"
    assert legacy_dataset._feature_names == ["col1", "col2"]
    assert legacy_dataset.name == "catalog.schema.table"
    assert legacy_dataset.digest == "test-digest"
