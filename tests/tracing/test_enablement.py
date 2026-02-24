"""
Tests for mlflow.tracing.enablement module
"""

from unittest import mock

import pytest

import mlflow
from mlflow.entities.trace_location import UCSchemaLocation, UcTablePrefixLocation
from mlflow.exceptions import MlflowException
from mlflow.tracing.enablement import (
    set_experiment_trace_location,
    unset_experiment_trace_location,
)

from tests.tracing.helper import skip_when_testing_trace_sdk


@pytest.fixture
def mock_databricks_tracking_uri():
    with mock.patch("mlflow.tracking.get_tracking_uri", return_value="databricks"):
        yield


@skip_when_testing_trace_sdk
def test_set_experiment_trace_location(mock_databricks_tracking_uri):
    experiment_id = mlflow.create_experiment("test_experiment")
    location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")
    sql_warehouse_id = "test-warehouse-id"

    with mock.patch("mlflow.tracing.client.TracingClient") as mock_client_class:
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        expected_location = UCSchemaLocation(
            catalog_name="test_catalog",
            schema_name="test_schema",
        )
        expected_location._otel_logs_table_name = "logs_table"
        expected_location._otel_spans_table_name = "spans_table"
        mock_client._set_experiment_trace_location.return_value = expected_location

        result = set_experiment_trace_location(
            location=location,
            experiment_id=experiment_id,
            sql_warehouse_id=sql_warehouse_id,
        )

        mock_client._set_experiment_trace_location.assert_called_once_with(
            location=location,
            experiment_id=experiment_id,
            sql_warehouse_id=sql_warehouse_id,
        )
        assert result == expected_location


def test_set_experiment_trace_location_with_default_experiment(mock_databricks_tracking_uri):
    location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")
    default_experiment_id = mlflow.set_experiment("test_experiment").experiment_id

    with (
        mock.patch("mlflow.tracing.client.TracingClient") as mock_client_class,
        mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value=default_experiment_id),
    ):
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        expected_location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")
        mock_client._set_experiment_trace_location.return_value = expected_location

        result = set_experiment_trace_location(location=location)
        mock_client._set_experiment_trace_location.assert_called_once_with(
            location=location,
            experiment_id=default_experiment_id,
            sql_warehouse_id=None,
        )

        assert result == expected_location


def test_set_experiment_trace_location_no_experiment(mock_databricks_tracking_uri):
    location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")
    with mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value=None):
        with pytest.raises(MlflowException, match="Experiment ID is required"):
            set_experiment_trace_location(location=location)


@skip_when_testing_trace_sdk
def test_set_experiment_trace_location_non_existent_experiment(mock_databricks_tracking_uri):
    location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")

    experiment_id = "12345"
    with pytest.raises(MlflowException, match="Could not find experiment with ID"):
        set_experiment_trace_location(location=location, experiment_id=experiment_id)


def test_unset_experiment_trace_location(mock_databricks_tracking_uri):
    experiment_id = "123"
    location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")

    with mock.patch("mlflow.tracing.client.TracingClient") as mock_client_class:
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client
        unset_experiment_trace_location(
            location=location,
            experiment_id=experiment_id,
        )
        mock_client._unset_experiment_trace_location.assert_called_once_with(
            experiment_id,
            location,
        )


def test_unset_experiment_trace_location_errors(mock_databricks_tracking_uri):
    with pytest.raises(MlflowException, match="must be an instance of"):
        unset_experiment_trace_location(location="test_catalog.test_schema")

    with mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value=None):
        with pytest.raises(MlflowException, match="Experiment ID is required"):
            unset_experiment_trace_location(
                location=UCSchemaLocation("test_catalog", "test_schema")
            )


def test_unset_experiment_trace_location_with_default_experiment(mock_databricks_tracking_uri):
    default_experiment_id = "456"

    with (
        mock.patch("mlflow.tracing.client.TracingClient") as mock_client_class,
        mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value=default_experiment_id),
    ):
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")
        unset_experiment_trace_location(location)

        mock_client._unset_experiment_trace_location.assert_called_once_with(
            default_experiment_id,
            location,
        )


def test_non_databricks_tracking_uri_errors():
    with pytest.raises(
        MlflowException,
        match="The `set_experiment_trace_location` API is only supported on Databricks.",
    ):
        set_experiment_trace_location(location=UCSchemaLocation("test_catalog", "test_schema"))

    with pytest.raises(
        MlflowException,
        match="The `unset_experiment_trace_location` API is only supported on Databricks.",
    ):
        unset_experiment_trace_location(location=UCSchemaLocation("test_catalog", "test_schema"))


@skip_when_testing_trace_sdk
def test_set_experiment_trace_location_with_uc_table_prefix(mock_databricks_tracking_uri):
    experiment_id = mlflow.create_experiment("test_experiment_prefix")
    location = UcTablePrefixLocation(
        catalog_name="test_catalog",
        schema_name="test_schema",
        table_prefix="myapp_",
    )
    sql_warehouse_id = "test-warehouse-id"

    with mock.patch("mlflow.tracing.client.TracingClient") as mock_client_class:
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        expected_location = UcTablePrefixLocation(
            catalog_name="test_catalog",
            schema_name="test_schema",
            table_prefix="myapp_",
            spans_table_name="test_catalog.test_schema.myapp_otel_spans",
            logs_table_name="test_catalog.test_schema.myapp_otel_logs",
            metrics_table_name="test_catalog.test_schema.myapp_otel_metrics",
        )
        mock_client._set_experiment_trace_location.return_value = expected_location

        result = set_experiment_trace_location(
            location=location,
            experiment_id=experiment_id,
            sql_warehouse_id=sql_warehouse_id,
        )

        mock_client._set_experiment_trace_location.assert_called_once_with(
            location=location,
            experiment_id=experiment_id,
            sql_warehouse_id=sql_warehouse_id,
        )
        assert result == expected_location
        assert result.spans_table_name == "test_catalog.test_schema.myapp_otel_spans"


def test_unset_experiment_trace_location_with_uc_table_prefix(mock_databricks_tracking_uri):
    experiment_id = "123"
    location = UcTablePrefixLocation(
        catalog_name="test_catalog",
        schema_name="test_schema",
        table_prefix="myapp_",
    )

    with mock.patch("mlflow.tracing.client.TracingClient") as mock_client_class:
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client
        unset_experiment_trace_location(
            location=location,
            experiment_id=experiment_id,
        )
        mock_client._unset_experiment_trace_location.assert_called_once_with(
            experiment_id,
            location,
        )
