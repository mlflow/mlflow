from types import SimpleNamespace
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException, RestException
from mlflow.store._unity_catalog.registry import volume_weights
from mlflow.store._unity_catalog.registry.volume_weights import (
    get_volume_artifacts_path,
    start_volume_weights_copy,
    wait_for_volume_weights_copy,
)

_MODULE = "mlflow.store._unity_catalog.registry.volume_weights"


def _status(model_state, failure_reason=None):
    status = {"state": model_state, "model_artifact": {"state": model_state}}
    if failure_reason:
        status["failure_reason"] = failure_reason
    return status


@pytest.fixture
def stage():
    with mock.patch(f"{_MODULE}.stage_model_for_databricks_model_serving") as stage:
        yield stage


@pytest.fixture
def poll():
    with mock.patch(f"{_MODULE}.get_databricks_model_serving_staging_status") as poll:
        yield poll


@pytest.fixture(autouse=True)
def no_sleep():
    with mock.patch.object(volume_weights.time, "sleep") as sleep:
        yield sleep


@pytest.mark.parametrize(
    "metadata",
    [None, {}, {"other": "value"}],
)
def test_get_volume_artifacts_path_returns_none_without_a_volume_path(metadata):
    assert get_volume_artifacts_path(SimpleNamespace(metadata=metadata)) is None


@pytest.mark.parametrize(
    "path",
    ["/Volumes/main/default/weights", "/Volumes/main/default/weights/qwen/v1"],
)
def test_get_volume_artifacts_path_returns_a_volume_path(path):
    model = SimpleNamespace(metadata={"volume_artifacts_path": path})
    assert get_volume_artifacts_path(model) == path


@pytest.mark.parametrize(
    "path",
    ["/dbfs/weights", "Volumes/main/default/weights", "/Volumes/main/default", 42],
)
def test_get_volume_artifacts_path_rejects_a_path_outside_a_volume(path):
    model = SimpleNamespace(metadata={"volume_artifacts_path": path})
    with pytest.raises(MlflowException, match="must be a Unity Catalog volume path") as e:
        get_volume_artifacts_path(model)
    assert e.value.error_code == "INVALID_PARAMETER_VALUE"


def test_start_volume_weights_copy_returns_the_staging_status(stage):
    stage.return_value = {"status": _status("IN_PROGRESS")}
    host_creds = object()

    assert start_volume_weights_copy("c.s.m", "3", host_creds) == _status("IN_PROGRESS")
    stage.assert_called_once_with("c.s.m", "3", host_creds=host_creds)


def test_start_volume_weights_copy_raises_with_the_server_reason_and_code(stage):
    stage.side_effect = RestException({
        "error_code": "BAD_REQUEST",
        "message": "it wasn't registered from a logged model.",
    })
    with pytest.raises(MlflowException, match="wasn't registered from a logged model") as e:
        start_volume_weights_copy("c.s.m", "3", None)
    assert e.value.error_code == "BAD_REQUEST"


def test_start_volume_weights_copy_maps_an_unknown_server_code_to_internal_error(stage):
    stage.side_effect = RestException({"error_code": "NOT_A_CLIENT_CODE", "message": "nope"})
    with pytest.raises(MlflowException, match="nope") as e:
        start_volume_weights_copy("c.s.m", "3", None)
    assert e.value.error_code == "INTERNAL_ERROR"


def test_start_volume_weights_copy_raises_when_the_server_reports_no_status(stage):
    stage.return_value = {}
    with pytest.raises(MlflowException, match="doesn't report when Databricks finishes copying"):
        start_volume_weights_copy("c.s.m", "3", None)


def test_wait_returns_once_the_model_weights_are_copied(stage, poll, no_sleep):
    poll.side_effect = [_status("IN_PROGRESS"), _status("SUCCEEDED")]

    wait_for_volume_weights_copy("c.s.m", "3", _status("IN_PROGRESS"), None)

    assert poll.call_count == 2
    assert no_sleep.call_count == 2
    stage.assert_not_called()


def test_wait_ignores_the_environment_artifact(stage, poll):
    status = {
        "state": "IN_PROGRESS",
        "model_artifact": {"state": "SUCCEEDED"},
        "environment_artifact": {"state": "IN_PROGRESS"},
    }
    wait_for_volume_weights_copy("c.s.m", "3", status, None)
    poll.assert_not_called()


@pytest.mark.parametrize("failed_state", ["FAILED", "NOT_STAGED"])
def test_wait_starts_a_failed_copy_again(stage, poll, failed_state):
    stage.return_value = {"status": _status("IN_PROGRESS")}
    poll.return_value = _status("SUCCEEDED")

    wait_for_volume_weights_copy("c.s.m", "3", _status(failed_state), None)

    stage.assert_called_once_with("c.s.m", "3", host_creds=None)


def test_wait_raises_with_the_reason_once_restarts_are_exhausted(stage, poll):
    stage.return_value = {"status": _status("FAILED", "requirements_invalid")}

    with pytest.raises(MlflowException, match=r"into the model version \(requirements_invalid\)"):
        wait_for_volume_weights_copy("c.s.m", "3", _status("FAILED", "requirements_invalid"), None)

    assert stage.call_count == volume_weights._MAX_COPY_RESTARTS
    poll.assert_not_called()


def test_wait_raises_when_the_copy_outlasts_the_timeout(stage, poll, monkeypatch):
    monkeypatch.setenv("MLFLOW_DATABRICKS_VOLUME_WEIGHTS_COPY_TIMEOUT", "30")
    poll.return_value = _status("IN_PROGRESS")
    with (
        mock.patch.object(volume_weights.time, "monotonic", side_effect=[0, 10, 20, 31]),
        pytest.raises(MlflowException, match="within 30 seconds"),
    ):
        wait_for_volume_weights_copy("c.s.m", "3", _status("IN_PROGRESS"), None)

    assert poll.call_count == 2
    stage.assert_not_called()
