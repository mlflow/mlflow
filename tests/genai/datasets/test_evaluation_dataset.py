import json
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from mlflow.data.dataset_source_registry import (
    get_dataset_source_from_json,
    register_dataset_source,
)
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.genai.datasets.databricks_evaluation_dataset_source import (
    DatabricksEvaluationDatasetSource,
    DatabricksUCTableDatasetSource,
)
from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset


def create_test_source_json(table_name: str = "main.default.testtable") -> str:
    """Create a JSON string source value consistent with Databricks managed evaluation datasets.

    This format matches the behavior of Databricks managed evaluation datasets as of July 2025.
    """
    return json.dumps({"table_name": table_name})


def create_mock_managed_dataset(source_value: Any) -> Mock:
    """Create a mock Databricks Agent Evaluation ManagedDataset for testing"""
    mock_dataset = Mock()
    mock_dataset.dataset_id = getattr(source_value, "dataset_id", "test-dataset-id")
    mock_dataset.name = getattr(source_value, "_table_name", "catalog.schema.table")
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


@pytest.fixture
def mock_managed_dataset() -> Mock:
    """Create a mock Databricks Agent Evaluation ManagedDataset for testing."""
    return create_mock_managed_dataset(create_test_source_json())


def create_dataset_with_source(source_value: Any) -> EvaluationDataset:
    """Factory function to create EvaluationDataset with specific source value."""
    mock_dataset = create_mock_managed_dataset(source_value)
    return EvaluationDataset(mock_dataset)


def test_evaluation_dataset_properties(mock_managed_dataset):
    dataset = EvaluationDataset(mock_managed_dataset)

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
    assert isinstance(dataset.source, DatabricksEvaluationDatasetSource)
    assert dataset.source.table_name == "catalog.schema.table"
    assert dataset.source.dataset_id == "test-dataset-id"


def test_evaluation_dataset_source_with_string_source():
    dataset = create_dataset_with_source("string-value")

    assert isinstance(dataset.source, DatabricksEvaluationDatasetSource)
    assert dataset.source.table_name == "catalog.schema.table"
    assert dataset.source.dataset_id == "test-dataset-id"


def test_evaluation_dataset_source_with_none():
    dataset = create_dataset_with_source(None)

    assert isinstance(dataset.source, DatabricksEvaluationDatasetSource)
    assert dataset.source.table_name == "catalog.schema.table"
    assert dataset.source.dataset_id == "test-dataset-id"


def test_evaluation_dataset_source_always_returns_databricks_evaluation_dataset_source():
    existing_source = DatabricksEvaluationDatasetSource(
        table_name="existing.table", dataset_id="existing-id"
    )
    dataset = create_dataset_with_source(existing_source)

    assert isinstance(dataset.source, DatabricksEvaluationDatasetSource)
    assert dataset.source.table_name == "existing.table"
    assert dataset.source.dataset_id == "existing-id"

    spark_source = SparkDatasetSource(table_name="spark.table")
    dataset = create_dataset_with_source(spark_source)

    assert isinstance(dataset.source, DatabricksEvaluationDatasetSource)
    assert dataset.source.table_name == "spark.table"
    assert dataset.source.dataset_id == "test-dataset-id"


def test_evaluation_dataset_to_df(mock_managed_dataset):
    dataset = EvaluationDataset(mock_managed_dataset)

    df = dataset.to_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    mock_managed_dataset.to_df.assert_called_once()


def test_evaluation_dataset_to_mlflow_entity(mock_managed_dataset):
    dataset = EvaluationDataset(mock_managed_dataset)

    entity = dataset._to_mlflow_entity()
    assert entity.name == "catalog.schema.table"
    assert entity.digest == "test-digest"
    assert entity.source_type == "databricks-uc-table"

    source_dict = json.loads(entity.source)
    assert source_dict["table_name"] == "catalog.schema.table"
    assert source_dict["dataset_id"] == "test-dataset-id"
    assert entity.schema == "test-schema"
    assert entity.profile == "test-profile"


def test_evaluation_dataset_to_mlflow_entity_with_existing_source():
    existing_source = DatabricksEvaluationDatasetSource(
        table_name="existing.table", dataset_id="existing-id"
    )
    dataset = create_dataset_with_source(existing_source)

    entity = dataset._to_mlflow_entity()
    assert entity.name == "existing.table"
    assert entity.digest == "test-digest"
    assert entity.source_type == "databricks-uc-table"

    source_dict = json.loads(entity.source)
    assert source_dict["table_name"] == "existing.table"
    assert source_dict["dataset_id"] == "existing-id"
    assert entity.schema == "test-schema"
    assert entity.profile == "test-profile"


def test_evaluation_dataset_set_profile(mock_managed_dataset):
    dataset = EvaluationDataset(mock_managed_dataset)

    new_dataset = dataset.set_profile("new-profile")
    assert isinstance(new_dataset, EvaluationDataset)
    mock_managed_dataset.set_profile.assert_called_once_with("new-profile")


def test_evaluation_dataset_merge_records(mock_managed_dataset):
    dataset = EvaluationDataset(mock_managed_dataset)

    new_records = [{"col1": 4, "col2": "d"}]
    new_dataset = dataset.merge_records(new_records)
    assert isinstance(new_dataset, EvaluationDataset)
    mock_managed_dataset.merge_records.assert_called_once_with(new_records)


def test_evaluation_dataset_digest_computation(mock_managed_dataset):
    # Test when managed dataset has no digest
    mock_managed_dataset.digest = None

    dataset = EvaluationDataset(mock_managed_dataset)
    digest = dataset.digest

    assert digest is not None


def test_evaluation_dataset_to_evaluation_dataset(mock_managed_dataset):
    dataset = EvaluationDataset(mock_managed_dataset)

    legacy_dataset = dataset.to_evaluation_dataset(
        path="/path/to/data", feature_names=["col1", "col2"]
    )

    assert legacy_dataset._features_data.equals(dataset.to_df())
    assert legacy_dataset._path == "/path/to/data"
    assert legacy_dataset._feature_names == ["col1", "col2"]
    assert legacy_dataset.name == "catalog.schema.table"
    assert legacy_dataset.digest == "test-digest"


def test_databricks_uc_table_dataset_source():
    register_dataset_source(DatabricksUCTableDatasetSource)

    source_json = json.dumps({"table_name": "catalog.schema.table", "dataset_id": "test-id"})

    source = get_dataset_source_from_json(source_json, "databricks-uc-table")
    assert isinstance(source, DatabricksUCTableDatasetSource)
    assert source._get_source_type() == "databricks-uc-table"
    assert source.table_name == "catalog.schema.table"
    assert source.dataset_id == "test-id"
