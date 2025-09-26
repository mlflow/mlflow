import pytest

from mlflow.exceptions import MlflowException
from mlflow.tracing.destination import Databricks, DatabricksUnityCatalog


def test_set_destination_databricks():
    destination = Databricks(experiment_id="123")
    assert destination.type == "databricks"
    assert destination.experiment_id == "123"

    destination = Databricks(experiment_id=123)
    assert destination.experiment_id == "123"

    destination = Databricks(experiment_name="Default")
    assert destination.experiment_name == "Default"
    assert destination.experiment_id == "0"

    destination = Databricks(experiment_name="Default", experiment_id="0")
    assert destination.experiment_id == "0"

    with pytest.raises(MlflowException, match=r"experiment_id and experiment_name must"):
        Databricks(experiment_name="Default", experiment_id="123")


def test_set_destination_databricks_unity_catalog():
    destination = DatabricksUnityCatalog(
        catalog_name="test_catalog",
        schema_name="test_schema",
        spans_table_name="spans_table",
    )
    assert destination.catalog_name == "test_catalog"
    assert destination.schema_name == "test_schema"
    assert destination.spans_table_name == "spans_table"
    assert destination.type == "databricks_unity_catalog"


def test_databricks_unity_catalog_full_spans_table_name():
    destination = DatabricksUnityCatalog(
        catalog_name="catalog",
        schema_name="schema",
        spans_table_name="spans",
    )
    assert destination.full_spans_table_name == "catalog.schema.spans"
