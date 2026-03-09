from unittest import mock

import pytest

from mlflow.entities import Experiment
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.entities.trace_location import UnityCatalog
from mlflow.exceptions import MlflowException
from mlflow.tracing.provider import _MLFLOW_TRACE_USER_DESTINATION
from mlflow.tracking.fluent import (
    _register_experiment_trace_location,
    _sync_trace_destination_and_provider,
)
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
        _register_experiment_trace_location(
            experiment=_experiment(),
            trace_location="not-a-location",
        )


def test_uc_schema_location_is_rejected():
    from mlflow.entities.trace_location import UCSchemaLocation

    with pytest.raises(MlflowException, match="UnityCatalog"):
        _register_experiment_trace_location(
            experiment=_experiment(),
            trace_location=UCSchemaLocation("catalog", "schema"),
        )


def test_no_trace_location_returns_none():
    result = _register_experiment_trace_location(
        experiment=_experiment(),
        trace_location=None,
    )
    assert result is None


def test_non_databricks_backend_raises():
    with (
        mock.patch("mlflow.tracking.fluent._resolve_tracking_uri", return_value="file:///tmp"),
        mock.patch("mlflow.tracking.fluent.is_databricks_uri", return_value=False),
    ):
        with pytest.raises(MlflowException, match="only supported with a Databricks tracking URI"):
            _register_experiment_trace_location(
                experiment=_experiment(),
                trace_location=UnityCatalog("catalog", "schema", "prefix"),
            )


def test_creates_and_links_when_no_existing_location():
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

        result = _register_experiment_trace_location(
            experiment=_experiment(),
            trace_location=requested,
        )

        assert result is resolved
        tc._get_trace_location.assert_not_called()
        tc._create_or_get_trace_location.assert_called_once_with(requested)
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

        result = _register_experiment_trace_location(
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
            _register_experiment_trace_location(
                experiment=experiment,
                trace_location=requested,
            )


def test_apply_state_sets_destination_when_resolved():
    experiment = _experiment()
    resolved = UnityCatalog("catalog", "schema", table_prefix="prefix")

    _MLFLOW_TRACE_USER_DESTINATION.reset()
    with (
        mock.patch("mlflow.tracing.provider._initialize_tracer_provider") as init_provider,
        mock.patch("mlflow.tracing.provider.is_tracing_enabled", return_value=True),
    ):
        _sync_trace_destination_and_provider(experiment=experiment, resolved_location=resolved)

    assert _MLFLOW_TRACE_USER_DESTINATION.get() is resolved
    init_provider.assert_called_once()
    _MLFLOW_TRACE_USER_DESTINATION.reset()


def test_apply_state_reinits_for_uc_linked_experiment():
    experiment = _experiment(tags={MLFLOW_EXPERIMENT_DATABRICKS_TELEMETRY_DESTINATION_ID: "uuid"})

    _MLFLOW_TRACE_USER_DESTINATION.reset()
    with (
        mock.patch("mlflow.tracing.provider._initialize_tracer_provider") as init_provider,
        mock.patch("mlflow.tracing.provider.is_tracing_enabled", return_value=True),
    ):
        _sync_trace_destination_and_provider(experiment=experiment, resolved_location=None)

    assert _MLFLOW_TRACE_USER_DESTINATION._global_value is None
    init_provider.assert_called_once()
    _MLFLOW_TRACE_USER_DESTINATION.reset()


def test_apply_state_clears_global_slot_on_experiment_switch():
    experiment = _experiment()
    old_dest = UnityCatalog("catalog", "schema", table_prefix="old")

    _MLFLOW_TRACE_USER_DESTINATION.reset()
    _MLFLOW_TRACE_USER_DESTINATION.set(old_dest)
    assert _MLFLOW_TRACE_USER_DESTINATION.get() is old_dest

    with (
        mock.patch("mlflow.tracing.provider._initialize_tracer_provider") as init_provider,
        mock.patch("mlflow.tracing.provider.is_tracing_enabled", return_value=True),
    ):
        _sync_trace_destination_and_provider(experiment=experiment, resolved_location=None)

    assert _MLFLOW_TRACE_USER_DESTINATION._global_value is None
    init_provider.assert_called_once()
    _MLFLOW_TRACE_USER_DESTINATION.reset()


def test_apply_state_skips_reinit_when_no_previous_destination():
    experiment = _experiment()

    _MLFLOW_TRACE_USER_DESTINATION.reset()
    with (
        mock.patch("mlflow.tracing.provider._initialize_tracer_provider") as init_provider,
        mock.patch("mlflow.tracing.provider.is_tracing_enabled", return_value=True),
    ):
        _sync_trace_destination_and_provider(experiment=experiment, resolved_location=None)

    init_provider.assert_not_called()
    _MLFLOW_TRACE_USER_DESTINATION.reset()


def test_apply_state_skips_reinit_for_uc_to_uc_switch():
    experiment = _experiment()
    old_dest = UnityCatalog("catalog", "schema", table_prefix="old")
    new_dest = UnityCatalog("catalog", "schema", table_prefix="new")

    _MLFLOW_TRACE_USER_DESTINATION.reset()
    _MLFLOW_TRACE_USER_DESTINATION.set(old_dest)

    with (
        mock.patch("mlflow.tracing.provider._initialize_tracer_provider") as init_provider,
        mock.patch("mlflow.tracing.provider.is_tracing_enabled", return_value=True),
    ):
        _sync_trace_destination_and_provider(experiment=experiment, resolved_location=new_dest)

    assert _MLFLOW_TRACE_USER_DESTINATION.get() is new_dest
    init_provider.assert_not_called()
    _MLFLOW_TRACE_USER_DESTINATION.reset()


def test_apply_state_skips_reinit_when_tracing_disabled():
    experiment = _experiment()
    old_dest = UnityCatalog("catalog", "schema", table_prefix="old")

    _MLFLOW_TRACE_USER_DESTINATION.reset()
    _MLFLOW_TRACE_USER_DESTINATION.set(old_dest)

    with (
        mock.patch("mlflow.tracing.provider._initialize_tracer_provider") as init_provider,
        mock.patch("mlflow.tracing.provider.is_tracing_enabled", return_value=False),
    ):
        _sync_trace_destination_and_provider(experiment=experiment, resolved_location=None)

    init_provider.assert_not_called()
    _MLFLOW_TRACE_USER_DESTINATION.reset()
