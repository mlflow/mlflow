import json

import pytest

from mlflow.genai.datasets.databricks_evaluation_dataset_source import (
    DatabricksEvaluationDatasetSource,
)


def test_databricks_evaluation_dataset_source_init():
    # Test with table_name
    source = DatabricksEvaluationDatasetSource(table_name="catalog.schema.table")
    assert source.table_name == "catalog.schema.table"
    assert source.dataset_id is None

    # Test with dataset_id
    source = DatabricksEvaluationDatasetSource(dataset_id="12345")
    assert source.table_name is None
    assert source.dataset_id == "12345"

    # Test with both
    source = DatabricksEvaluationDatasetSource(
        table_name="catalog.schema.table", dataset_id="12345"
    )
    assert source.table_name == "catalog.schema.table"
    assert source.dataset_id == "12345"

    # Test without either should raise ValueError
    with pytest.raises(ValueError, match="Either table_name or dataset_id must be provided"):
        DatabricksEvaluationDatasetSource()


def test_databricks_evaluation_dataset_source_get_source_type():
    assert DatabricksEvaluationDatasetSource._get_source_type() == "databricks_evaluation_dataset"


def test_databricks_evaluation_dataset_source_to_dict():
    # Test with table_name only
    source = DatabricksEvaluationDatasetSource(table_name="catalog.schema.table")
    assert source.to_dict() == {"table_name": "catalog.schema.table"}

    # Test with dataset_id only
    source = DatabricksEvaluationDatasetSource(dataset_id="12345")
    assert source.to_dict() == {"dataset_id": "12345"}

    # Test with both
    source = DatabricksEvaluationDatasetSource(
        table_name="catalog.schema.table", dataset_id="12345"
    )
    assert source.to_dict() == {
        "table_name": "catalog.schema.table",
        "dataset_id": "12345",
    }


def test_databricks_evaluation_dataset_source_from_dict():
    # Test with table_name only
    source_dict = {"table_name": "catalog.schema.table"}
    source = DatabricksEvaluationDatasetSource.from_dict(source_dict)
    assert source.table_name == "catalog.schema.table"
    assert source.dataset_id is None

    # Test with dataset_id only
    source_dict = {"dataset_id": "12345"}
    source = DatabricksEvaluationDatasetSource.from_dict(source_dict)
    assert source.table_name is None
    assert source.dataset_id == "12345"

    # Test with both
    source_dict = {"table_name": "catalog.schema.table", "dataset_id": "12345"}
    source = DatabricksEvaluationDatasetSource.from_dict(source_dict)
    assert source.table_name == "catalog.schema.table"
    assert source.dataset_id == "12345"


def test_databricks_evaluation_dataset_source_to_json():
    source = DatabricksEvaluationDatasetSource(
        table_name="catalog.schema.table", dataset_id="12345"
    )
    json_str = source.to_json()
    parsed = json.loads(json_str)
    assert parsed == {"table_name": "catalog.schema.table", "dataset_id": "12345"}


def test_databricks_evaluation_dataset_source_from_json():
    json_str = json.dumps({"table_name": "catalog.schema.table", "dataset_id": "12345"})
    source = DatabricksEvaluationDatasetSource.from_json(json_str)
    assert source.table_name == "catalog.schema.table"
    assert source.dataset_id == "12345"


def test_databricks_evaluation_dataset_source_load_not_implemented():
    source = DatabricksEvaluationDatasetSource(table_name="catalog.schema.table")
    with pytest.raises(
        NotImplementedError,
        match="Loading a Databricks Evaluation Dataset from source is not supported",
    ):
        source.load()


def test_databricks_evaluation_dataset_source_can_resolve():
    # _can_resolve should return False for all inputs
    assert DatabricksEvaluationDatasetSource._can_resolve({}) is False
    assert DatabricksEvaluationDatasetSource._can_resolve({"table_name": "test"}) is False


def test_databricks_evaluation_dataset_source_resolve_not_implemented():
    with pytest.raises(
        NotImplementedError, match="Resolution from a source dictionary is not supported"
    ):
        DatabricksEvaluationDatasetSource._resolve({})
