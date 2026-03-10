from unittest import mock

import pytest

import mlflow
from mlflow.entities import Experiment
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.entities.trace_location import UnityCatalog
from mlflow.exceptions import MlflowException
from mlflow.tracking.fluent import _resolve_experiment_to_trace_location
from mlflow.utils.mlflow_tags import MLFLOW_EXPERIMENT_DATABRICKS_TELEMETRY_DESTINATION_ID


def _experiment(tags=None):
    tag_entities = [ExperimentTag(k, v) for k, v in (tags or {}).items()]
    return Experiment(
        experiment_id="123",
        name="test-experiment",
        artifact_location="file:/tmp",
        lifecycle_stage="active",
        tags=tag_entities,
    )


def test_invalid_type_raises():
    with pytest.raises(MlflowException, match="UnityCatalog"):
        _resolve_experiment_to_trace_location(
            experiment=_experiment(),
            trace_location="not-a-location",
        )


def test_uc_schema_location_is_rejected():
    from mlflow.entities.trace_location import UCSchemaLocation

    with pytest.raises(MlflowException, match="UnityCatalog"):
        _resolve_experiment_to_trace_location(
            experiment=_experiment(),
            trace_location=UCSchemaLocation("catalog", "schema"),
        )


def test_no_trace_location_returns_none():
    with mock.patch("mlflow.tracking.fluent.TracingClient") as mock_tc_cls:
        result = _resolve_experiment_to_trace_location(
            experiment=_experiment(),
            trace_location=None,
        )
        assert result is None
        mock_tc_cls.assert_not_called()


def test_non_databricks_backend_raises():
    with (
        mock.patch("mlflow.tracking.fluent._resolve_tracking_uri", return_value="file:///tmp"),
        mock.patch("mlflow.tracking.fluent.is_databricks_uri", return_value=False),
    ):
        with pytest.raises(MlflowException, match="only supported with a Databricks tracking URI"):
            _resolve_experiment_to_trace_location(
                experiment=_experiment(),
                trace_location=UnityCatalog("catalog", "schema", "prefix"),
            )


def test_set_experiment_with_table_prefix_env_var_points_to_trace_location_param(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACING_DESTINATION", "catalog.schema.prefix")

    with pytest.raises(
        MlflowException,
        match=r"Unity Catalog table-prefix destinations "
        r"\(<catalog_name>\.<schema_name>\.<table_prefix>\) are not supported in "
        r"MLFLOW_TRACING_DESTINATION.*Use `mlflow\.set_experiment",
    ):
        mlflow.set_experiment("test-experiment")


def test_creates_and_links_when_no_existing_location(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", "warehouse-1")
    requested = UnityCatalog("catalog", "schema", table_prefix="prefix")
    resolved = UnityCatalog("catalog", "schema", table_prefix="prefix")

    with (
        mock.patch("mlflow.tracking.fluent._resolve_tracking_uri", return_value="databricks"),
        mock.patch("mlflow.tracking.fluent.is_databricks_uri", return_value=True),
        mock.patch("mlflow.tracking.fluent.TracingClient") as tc_cls,
    ):
        tc = tc_cls.return_value
        tc._get_trace_location.return_value = None
        tc._create_or_get_trace_location.return_value = resolved

        result = _resolve_experiment_to_trace_location(
            experiment=_experiment(),
            trace_location=requested,
        )

        assert result is resolved
        tc._get_trace_location.assert_not_called()
        tc._create_or_get_trace_location.assert_called_once_with(requested, "warehouse-1")
        tc._link_trace_location.assert_called_once_with(
            experiment_id="123",
            location=resolved,
        )


def test_noop_when_existing_location_matches():
    requested = UnityCatalog("catalog", "schema", table_prefix="prefix")
    existing = UnityCatalog("catalog", "schema", table_prefix="prefix")
    experiment = _experiment(
        tags={MLFLOW_EXPERIMENT_DATABRICKS_TELEMETRY_DESTINATION_ID: "some-uuid"}
    )

    with (
        mock.patch("mlflow.tracking.fluent._resolve_tracking_uri", return_value="databricks"),
        mock.patch("mlflow.tracking.fluent.is_databricks_uri", return_value=True),
        mock.patch("mlflow.tracking.fluent.TracingClient") as tc_cls,
    ):
        tc = tc_cls.return_value
        tc._get_trace_location.return_value = existing

        result = _resolve_experiment_to_trace_location(
            experiment=experiment,
            trace_location=requested,
        )

        assert result is existing
        tc._get_trace_location.assert_called_once_with("some-uuid")
        tc._create_or_get_trace_location.assert_not_called()
        tc._link_trace_location.assert_not_called()


def test_errors_when_existing_location_differs():
    requested = UnityCatalog("catalog", "schema", table_prefix="new_prefix")
    existing = UnityCatalog("catalog", "schema", table_prefix="old_prefix")
    experiment = _experiment(
        tags={MLFLOW_EXPERIMENT_DATABRICKS_TELEMETRY_DESTINATION_ID: "some-uuid"}
    )

    with (
        mock.patch("mlflow.tracking.fluent._resolve_tracking_uri", return_value="databricks"),
        mock.patch("mlflow.tracking.fluent.is_databricks_uri", return_value=True),
        mock.patch("mlflow.tracking.fluent.TracingClient") as tc_cls,
    ):
        tc = tc_cls.return_value
        tc._get_trace_location.return_value = existing

        with pytest.raises(MlflowException, match="already linked to a different"):
            _resolve_experiment_to_trace_location(
                experiment=experiment,
                trace_location=requested,
            )


def test_set_experiment_wires_trace_location_to_returned_experiment():
    resolved = UnityCatalog("catalog", "schema", table_prefix="pfx")

    with (
        mock.patch(
            "mlflow.tracking.fluent._resolve_experiment_to_trace_location",
            return_value=resolved,
        ) as mock_register,
        mock.patch(
            "mlflow.tracking.fluent._sync_trace_destination_and_provider",
        ) as mock_sync,
    ):
        experiment = mlflow.set_experiment("test-trace-loc-integration")

    mock_register.assert_called_once()
    _, kwargs = mock_register.call_args
    assert kwargs["experiment"].name == "test-trace-loc-integration"
    mock_sync.assert_called_once_with(resolved, experiment)
    assert experiment.trace_location is resolved


def test_set_experiment_with_trace_location_installs_uc_processor():
    from mlflow.tracing.export.uc_table import DatabricksUCTableSpanExporter
    from mlflow.tracing.processor.uc_table import DatabricksUCTableSpanProcessor
    from mlflow.tracing.provider import _MLFLOW_TRACE_USER_DESTINATION, _get_tracer

    resolved = UnityCatalog("catalog", "schema", table_prefix="pfx")
    mlflow.tracing.reset()
    _MLFLOW_TRACE_USER_DESTINATION.reset()

    with (
        mock.patch(
            "mlflow.tracking.fluent._resolve_experiment_to_trace_location",
            return_value=resolved,
        ) as mock_register,
    ):
        experiment = mlflow.set_experiment("test-uc-processor")

    mock_register.assert_called_once()
    assert experiment.trace_location is resolved

    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], DatabricksUCTableSpanProcessor)
    assert isinstance(processors[0].span_exporter, DatabricksUCTableSpanExporter)

    _MLFLOW_TRACE_USER_DESTINATION.reset()
    mlflow.tracing.reset()


def test_set_experiment_without_trace_location_does_not_install_uc_processor():
    from mlflow.tracing.processor.uc_table import DatabricksUCTableSpanProcessor
    from mlflow.tracing.provider import _MLFLOW_TRACE_USER_DESTINATION, _get_tracer

    mlflow.tracing.reset()
    _MLFLOW_TRACE_USER_DESTINATION.reset()

    mlflow.set_experiment("test-no-uc-processor")

    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors
    assert all(not isinstance(p, DatabricksUCTableSpanProcessor) for p in processors)

    _MLFLOW_TRACE_USER_DESTINATION.reset()
    mlflow.tracing.reset()
