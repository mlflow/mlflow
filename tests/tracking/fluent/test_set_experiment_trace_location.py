from unittest import mock

import pytest

from mlflow.entities import Experiment
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.entities.trace_location import UnityCatalog
from mlflow.exceptions import MlflowException
from mlflow.tracking.fluent import (
    _apply_experiment_trace_destination_state,
    _resolve_experiment_trace_destination,
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
        _resolve_experiment_trace_destination(
            experiment=_experiment(),
            trace_location="not-a-location",
            client=mock.MagicMock(),
        )


def test_uc_schema_location_is_rejected():
    from mlflow.entities.trace_location import UCSchemaLocation

    with pytest.raises(MlflowException, match="UnityCatalog"):
        _resolve_experiment_trace_destination(
            experiment=_experiment(),
            trace_location=UCSchemaLocation("catalog", "schema"),
            client=mock.MagicMock(),
        )


def test_no_trace_location_returns_none():
    result = _resolve_experiment_trace_destination(
        experiment=_experiment(),
        trace_location=None,
        client=mock.MagicMock(),
    )
    assert result is None


def test_non_databricks_backend_raises():
    with (
        mock.patch("mlflow.tracking.fluent._resolve_tracking_uri", return_value="file:///tmp"),
        mock.patch("mlflow.tracking.fluent.is_databricks_uri", return_value=False),
    ):
        with pytest.raises(MlflowException, match="only supported with a Databricks tracking URI"):
            _resolve_experiment_trace_destination(
                experiment=_experiment(),
                trace_location=UnityCatalog("catalog", "schema", "prefix"),
                client=mock.MagicMock(),
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

        result = _resolve_experiment_trace_destination(
            experiment=_experiment(),
            trace_location=requested,
            client=mock.MagicMock(),
        )

        assert result is resolved
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

        result = _resolve_experiment_trace_destination(
            experiment=experiment,
            trace_location=requested,
            client=mock.MagicMock(),
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
            _resolve_experiment_trace_destination(
                experiment=experiment,
                trace_location=requested,
                client=mock.MagicMock(),
            )


def test_apply_state_sets_destination_when_resolved():
    experiment = _experiment()
    resolved = UnityCatalog("catalog", "schema", table_prefix="prefix")

    with (
        mock.patch("mlflow.tracking.fluent._set_experiment_derived_destination") as set_dest,
        mock.patch("mlflow.tracking.fluent._mark_provider_for_reinit") as mark_reinit,
        mock.patch("mlflow.tracking.fluent._clear_experiment_derived_destination") as clear_dest,
    ):
        _apply_experiment_trace_destination_state(experiment=experiment, resolved_location=resolved)

        set_dest.assert_called_once_with(resolved)
        mark_reinit.assert_not_called()
        clear_dest.assert_not_called()
        assert experiment.trace_location is resolved


def test_apply_state_marks_reinit_for_uc_linked_experiment():
    experiment = _experiment(tags={MLFLOW_EXPERIMENT_DATABRICKS_TELEMETRY_DESTINATION_ID: "uuid"})

    with (
        mock.patch("mlflow.tracking.fluent._set_experiment_derived_destination") as set_dest,
        mock.patch("mlflow.tracking.fluent._mark_provider_for_reinit") as mark_reinit,
        mock.patch("mlflow.tracking.fluent._clear_experiment_derived_destination") as clear_dest,
        mock.patch(
            "mlflow.tracking.fluent._get_experiment_derived_destination_experiment_id",
            return_value=None,
        ),
    ):
        _apply_experiment_trace_destination_state(experiment=experiment, resolved_location=None)

        mark_reinit.assert_called_once_with()
        set_dest.assert_not_called()
        clear_dest.assert_not_called()


def test_apply_state_clears_stale_cached_destination_for_non_uc_experiment():
    experiment = _experiment()

    with (
        mock.patch("mlflow.tracking.fluent._set_experiment_derived_destination") as set_dest,
        mock.patch("mlflow.tracking.fluent._mark_provider_for_reinit") as mark_reinit,
        mock.patch("mlflow.tracking.fluent._clear_experiment_derived_destination") as clear_dest,
        mock.patch(
            "mlflow.tracking.fluent._get_experiment_derived_destination_experiment_id",
            return_value="different-experiment",
        ),
    ):
        _apply_experiment_trace_destination_state(experiment=experiment, resolved_location=None)

        clear_dest.assert_called_once_with()
        set_dest.assert_not_called()
        mark_reinit.assert_not_called()


def test_apply_state_noop_for_non_uc_experiment_without_stale_cache():
    experiment = _experiment()

    with (
        mock.patch("mlflow.tracking.fluent._set_experiment_derived_destination") as set_dest,
        mock.patch("mlflow.tracking.fluent._mark_provider_for_reinit") as mark_reinit,
        mock.patch("mlflow.tracking.fluent._clear_experiment_derived_destination") as clear_dest,
        mock.patch(
            "mlflow.tracking.fluent._get_experiment_derived_destination_experiment_id",
            return_value=experiment.experiment_id,
        ),
    ):
        _apply_experiment_trace_destination_state(experiment=experiment, resolved_location=None)

        set_dest.assert_not_called()
        mark_reinit.assert_not_called()
        clear_dest.assert_not_called()
