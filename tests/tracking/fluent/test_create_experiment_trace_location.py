from unittest import mock

import pytest

import mlflow
import mlflow.tracking.fluent as fluent_module
from mlflow.entities import Experiment
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.entities.trace_location import UnityCatalog
from mlflow.exceptions import MlflowException


def _experiment(experiment_id="789"):
    return Experiment(
        experiment_id=experiment_id,
        name="test-experiment",
        artifact_location="file:/tmp",
        lifecycle_stage="active",
        tags=[ExperimentTag("key", "val")],
    )


def test_without_trace_location_unchanged():
    with mock.patch("mlflow.tracking.fluent.MlflowClient") as mock_client_cls:
        client = mock_client_cls.return_value
        client.create_experiment.return_value = "789"

        assert mlflow.create_experiment("my-exp", tags={"k": "v"}) == "789"
        client.create_experiment.assert_called_once_with("my-exp", None, {"k": "v"})
        client.get_experiment.assert_not_called()


def test_defaults_empty_prefix_to_experiment_id():
    with (
        mock.patch("mlflow.tracking.fluent.MlflowClient") as mock_client_cls,
        mock.patch(
            "mlflow.tracking.fluent._resolve_experiment_to_trace_location",
            return_value=None,
        ) as mock_resolve,
    ):
        client = mock_client_cls.return_value
        client.create_experiment.return_value = "789"
        client.get_experiment.return_value = _experiment()

        original = UnityCatalog("catalog", "schema")
        mlflow.create_experiment("my-exp", trace_location=original)

        passed_location = mock_resolve.call_args.kwargs["trace_location"]
        assert passed_location.table_prefix == "789"
        assert original.table_prefix is None


def test_explicit_prefix_preserved():
    with (
        mock.patch("mlflow.tracking.fluent.MlflowClient") as mock_client_cls,
        mock.patch(
            "mlflow.tracking.fluent._resolve_experiment_to_trace_location",
            return_value=None,
        ) as mock_resolve,
    ):
        client = mock_client_cls.return_value
        client.create_experiment.return_value = "789"
        client.get_experiment.return_value = _experiment()

        mlflow.create_experiment(
            "my-exp",
            trace_location=UnityCatalog("catalog", "schema", "custom_prefix"),
        )

        assert mock_resolve.call_args.kwargs["trace_location"].table_prefix == "custom_prefix"


def test_resolve_receives_experiment_object():
    exp = _experiment()

    with (
        mock.patch("mlflow.tracking.fluent.MlflowClient") as mock_client_cls,
        mock.patch(
            "mlflow.tracking.fluent._resolve_experiment_to_trace_location",
            return_value=None,
        ) as mock_resolve,
    ):
        client = mock_client_cls.return_value
        client.create_experiment.return_value = "789"
        client.get_experiment.return_value = exp

        result = mlflow.create_experiment(
            "my-exp",
            trace_location=UnityCatalog("catalog", "schema", "pfx"),
        )

        assert result == "789"
        assert mock_resolve.call_args.kwargs["experiment"] is exp


def test_does_not_sync_provider_or_set_active():
    original_active = fluent_module._active_experiment_id

    with (
        mock.patch("mlflow.tracking.fluent.MlflowClient") as mock_client_cls,
        mock.patch(
            "mlflow.tracking.fluent._resolve_experiment_to_trace_location",
            return_value=None,
        ),
        mock.patch(
            "mlflow.tracking.fluent._sync_trace_destination_and_provider",
        ) as mock_sync,
    ):
        client = mock_client_cls.return_value
        client.create_experiment.return_value = "789"
        client.get_experiment.return_value = _experiment()

        mlflow.create_experiment(
            "my-exp",
            trace_location=UnityCatalog("catalog", "schema", "pfx"),
        )

        mock_sync.assert_not_called()
        assert fluent_module._active_experiment_id == original_active


def test_link_failure_includes_retry_guidance():
    with (
        mock.patch("mlflow.tracking.fluent.MlflowClient") as mock_client_cls,
        mock.patch(
            "mlflow.tracking.fluent._resolve_experiment_to_trace_location",
            side_effect=MlflowException("backend error"),
        ),
    ):
        client = mock_client_cls.return_value
        client.create_experiment.return_value = "456"
        client.get_experiment.return_value = _experiment(experiment_id="456")

        with pytest.raises(MlflowException, match="delete the experiment and retry") as exc_info:
            mlflow.create_experiment(
                "new-exp",
                trace_location=UnityCatalog("cat", "sch", "pfx"),
            )

        assert "was created" in exc_info.value.message
        assert "backend error" in exc_info.value.message
