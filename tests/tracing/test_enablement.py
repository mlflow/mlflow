"""
Tests for mlflow.tracing.enablement module
"""

from unittest import mock

import pytest

from mlflow.entities.trace_location import UCSchemaLocation
from mlflow.exceptions import MlflowException
from mlflow.tracing.enablement import (
    set_experiment_trace_location,
    unset_experiment_trace_location,
)


@pytest.fixture
def mock_databricks_tracking_uri():
    with mock.patch("mlflow.tracing.enablement.get_tracking_uri", return_value="databricks"):
        yield


def test_set_experiment_trace_location(mock_databricks_tracking_uri):
    experiment_id = "123"
    location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")
    sql_warehouse_id = "test-warehouse-id"

    # Mock the TracingClient and its method
    with mock.patch("mlflow.tracing.enablement.TracingClient") as mock_client_class:
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the return value of _set_experiment_trace_location
        expected_location = UCSchemaLocation(
            catalog_name="test_catalog",
            schema_name="test_schema",
            otel_spans_table_name="spans_table",
            otel_logs_table_name="logs_table",
        )
        mock_client._set_experiment_trace_location.return_value = expected_location

        # Test with explicit experiment ID and sql_warehouse_id
        result = set_experiment_trace_location(
            location=location,
            experiment_id=experiment_id,
            sql_warehouse_id=sql_warehouse_id,
        )

        # Verify the correct method was called with correct arguments
        mock_client._set_experiment_trace_location.assert_called_once_with(
            uc_schema=location,
            experiment_id=experiment_id,
            sql_warehouse_id=sql_warehouse_id,
        )

        # Verify the return value
        assert result == expected_location


def test_set_experiment_trace_location_with_default_experiment(mock_databricks_tracking_uri):
    location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")
    default_experiment_id = "456"

    # Mock the TracingClient and _get_experiment_id
    with (
        mock.patch("mlflow.tracing.enablement.TracingClient") as mock_client_class,
        mock.patch(
            "mlflow.tracing.enablement._get_experiment_id", return_value=default_experiment_id
        ),
    ):
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        expected_location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")
        mock_client._set_experiment_trace_location.return_value = expected_location

        # Test without explicit experiment ID (uses default)
        result = set_experiment_trace_location(location=location)

        # Verify the method was called with the default experiment ID
        mock_client._set_experiment_trace_location.assert_called_once_with(
            uc_schema=location,
            experiment_id=default_experiment_id,
            sql_warehouse_id=None,
        )

        assert result == expected_location


def test_set_experiment_trace_location_no_experiment(mock_databricks_tracking_uri):
    location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")

    # Mock _get_experiment_id to return None
    with mock.patch("mlflow.tracing.enablement._get_experiment_id", return_value=None):
        with pytest.raises(MlflowException, match="Experiment ID is required"):
            set_experiment_trace_location(location=location)


def test_unset_experiment_trace_location(mock_databricks_tracking_uri):
    experiment_id = "123"
    location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")

    # Mock the TracingClient
    with mock.patch("mlflow.tracing.enablement.TracingClient") as mock_client_class:
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        # Test with explicit experiment ID and location
        unset_experiment_trace_location(
            location=location,
            experiment_id=experiment_id,
        )

        # Verify the correct method was called with correct arguments
        mock_client._unset_experiment_trace_location.assert_called_once_with(
            experiment_id,
            location.schema_location,
        )


def test_unset_experiment_trace_location_errors(mock_databricks_tracking_uri):
    with pytest.raises(MlflowException, match="must be an instance of"):
        unset_experiment_trace_location(location="test_catalog.test_schema")

    with mock.patch("mlflow.tracing.enablement._get_experiment_id", return_value=None):
        with pytest.raises(MlflowException, match="Experiment ID is required"):
            unset_experiment_trace_location(
                location=UCSchemaLocation("test_catalog", "test_schema")
            )


def test_unset_experiment_trace_location_with_default_experiment(mock_databricks_tracking_uri):
    default_experiment_id = "456"

    # Mock the TracingClient and _get_experiment_id
    with (
        mock.patch("mlflow.tracing.enablement.TracingClient") as mock_client_class,
        mock.patch(
            "mlflow.tracing.enablement._get_experiment_id", return_value=default_experiment_id
        ),
    ):
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client

        location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")
        # Test without explicit experiment ID (uses default)
        unset_experiment_trace_location(location)

        # Verify the method was called with the default experiment ID
        mock_client._unset_experiment_trace_location.assert_called_once_with(
            default_experiment_id,
            location.schema_location,
        )


def test_non_databricks_tracking_uri_errors():
    with pytest.raises(
        MlflowException,
        match="Setting storage location is only supported on Databricks Tracking Server.",
    ):
        set_experiment_trace_location(location=UCSchemaLocation("test_catalog", "test_schema"))

    with pytest.raises(
        MlflowException,
        match="Clearing storage location is only supported on Databricks Tracking Server.",
    ):
        unset_experiment_trace_location(location=UCSchemaLocation("test_catalog", "test_schema"))
