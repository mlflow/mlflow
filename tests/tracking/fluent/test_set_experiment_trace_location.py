from unittest import mock

import pytest

from mlflow.entities import Experiment
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.entities.trace_location import UnityCatalog
from mlflow.exceptions import MlflowException
from mlflow.tracking.fluent import _configure_experiment_trace_destination

_DEST_TAG = "mlflow.experiment.databricksTelemetryDestinationId"


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
        _configure_experiment_trace_destination(
            experiment=_experiment(),
            trace_location="not-a-location",
            client=mock.MagicMock(),
        )


def test_uc_schema_location_is_rejected():
    from mlflow.entities.trace_location import UCSchemaLocation

    with pytest.raises(MlflowException, match="UnityCatalog"):
        _configure_experiment_trace_destination(
            experiment=_experiment(),
            trace_location=UCSchemaLocation("catalog", "schema"),
            client=mock.MagicMock(),
        )


def test_no_trace_location_clears_destination():
    with (
        mock.patch("mlflow.tracking.fluent._clear_experiment_derived_destination") as clear_dest,
        mock.patch("mlflow.tracking.fluent._set_experiment_derived_destination") as set_dest,
    ):
        _configure_experiment_trace_destination(
            experiment=_experiment(),
            trace_location=None,
            client=mock.MagicMock(),
        )

        set_dest.assert_not_called()
        clear_dest.assert_called_once()


def test_creates_and_links_when_no_existing_location():
    requested = UnityCatalog("catalog", "schema", table_prefix="prefix")
    resolved = UnityCatalog("catalog", "schema", table_prefix="prefix")

    with (
        mock.patch("mlflow.tracking.fluent._resolve_tracking_uri", return_value="databricks"),
        mock.patch("mlflow.tracking.fluent.is_databricks_uri", return_value=True),
        mock.patch("mlflow.tracking.fluent._set_experiment_derived_destination") as set_dest,
        mock.patch("mlflow.tracking.fluent.TracingClient") as tc_cls,
    ):
        tc = tc_cls.return_value
        tc._get_trace_location.return_value = None
        tc._create_or_get_trace_location.return_value = resolved

        _configure_experiment_trace_destination(
            experiment=_experiment(),
            trace_location=requested,
            client=mock.MagicMock(),
        )

        tc._create_or_get_trace_location.assert_called_once_with(requested)
        tc._link_trace_location.assert_called_once_with(
            experiment_id="123",
            location=resolved,
        )
        set_dest.assert_called_once_with(resolved)


def test_noop_when_existing_location_matches():
    requested = UnityCatalog("catalog", "schema", table_prefix="prefix")
    existing = UnityCatalog("catalog", "schema", table_prefix="prefix")
    experiment = _experiment(tags={_DEST_TAG: "some-uuid"})

    with (
        mock.patch("mlflow.tracking.fluent._resolve_tracking_uri", return_value="databricks"),
        mock.patch("mlflow.tracking.fluent.is_databricks_uri", return_value=True),
        mock.patch("mlflow.tracking.fluent._set_experiment_derived_destination") as set_dest,
        mock.patch("mlflow.tracking.fluent.TracingClient") as tc_cls,
    ):
        tc = tc_cls.return_value
        tc._get_trace_location.return_value = existing

        _configure_experiment_trace_destination(
            experiment=experiment,
            trace_location=requested,
            client=mock.MagicMock(),
        )

        tc._get_trace_location.assert_called_once_with("some-uuid")
        tc._create_or_get_trace_location.assert_not_called()
        tc._link_trace_location.assert_not_called()
        set_dest.assert_called_once_with(existing)


def test_errors_when_existing_location_differs():
    requested = UnityCatalog("catalog", "schema", table_prefix="new_prefix")
    existing = UnityCatalog("catalog", "schema", table_prefix="old_prefix")
    experiment = _experiment(tags={_DEST_TAG: "some-uuid"})

    with (
        mock.patch("mlflow.tracking.fluent._resolve_tracking_uri", return_value="databricks"),
        mock.patch("mlflow.tracking.fluent.is_databricks_uri", return_value=True),
        mock.patch("mlflow.tracking.fluent.TracingClient") as tc_cls,
    ):
        tc = tc_cls.return_value
        tc._get_trace_location.return_value = existing

        with pytest.raises(MlflowException, match="already linked to a different"):
            _configure_experiment_trace_destination(
                experiment=experiment,
                trace_location=requested,
                client=mock.MagicMock(),
            )
