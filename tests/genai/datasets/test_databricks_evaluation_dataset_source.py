import json

import pytest

from mlflow.genai.datasets.databricks_evaluation_dataset_source import (
    DatabricksEvaluationDatasetSource,
)


def test_databricks_evaluation_dataset_source_init():
    source = DatabricksEvaluationDatasetSource(
        table_name="catalog.schema.table", dataset_id="12345"
    )
    assert source.table_name == "catalog.schema.table"
    assert source.dataset_id == "12345"


def test_databricks_evaluation_dataset_source_get_source_type():
    assert DatabricksEvaluationDatasetSource._get_source_type() == "databricks_evaluation_dataset"


def test_databricks_evaluation_dataset_source_to_dict():
    source = DatabricksEvaluationDatasetSource(
        table_name="catalog.schema.table", dataset_id="12345"
    )
    assert source.to_dict() == {
        "table_name": "catalog.schema.table",
        "dataset_id": "12345",
    }


def test_databricks_evaluation_dataset_source_from_dict():
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
    source = DatabricksEvaluationDatasetSource(
        table_name="catalog.schema.table", dataset_id="12345"
    )
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
