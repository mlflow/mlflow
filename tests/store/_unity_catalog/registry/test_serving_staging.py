from types import SimpleNamespace
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException, RestException
from mlflow.store._unity_catalog.registry import serving_staging
from mlflow.store._unity_catalog.registry.serving_staging import (
    get_staging_timeout,
    get_volume_artifacts_path,
    start_model_serving_staging,
    wait_for_model_artifact_staging,
)

_MODULE = "mlflow.store._unity_catalog.registry.serving_staging"


def _status(model_state):
    return {"state": model_state, "model_artifact": {"state": model_state}}


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
    with mock.patch.object(serving_staging.time, "sleep") as sleep:
        yield sleep


@pytest.mark.parametrize(
    "metadata",
    [None, {}, {"other": "value"}, ["not", "a", "dict"]],
)
def test_get_volume_artifacts_path_returns_none_without_a_volume_path(metadata):
    assert get_volume_artifacts_path(SimpleNamespace(metadata=metadata)) is None


@pytest.mark.parametrize(
    "path",
    ["/Volumes/main/default/weights", "/Volumes/main/default/weights/qwen/v1", "/Volumes/c/s/v/"],
)
def test_get_volume_artifacts_path_returns_a_volume_path(path):
    model = SimpleNamespace(metadata={"volume_artifacts_path": path})
    assert get_volume_artifacts_path(model) == path


@pytest.mark.parametrize(
    "path",
    [
        "/dbfs/weights",
        "Volumes/main/default/weights",
        "/Volumes/main/default",
        "/Volumes/main/default/weights/..",
        "/Volumes/main//default/weights",
        "/Volumes/main/./default/weights",
        "",
        42,
    ],
)
def test_get_volume_artifacts_path_rejects_a_path_outside_a_volume(path):
    model = SimpleNamespace(metadata={"volume_artifacts_path": path})
    with pytest.raises(MlflowException, match="must be a Unity Catalog volume path") as e:
        get_volume_artifacts_path(model)
    assert e.value.error_code == "INVALID_PARAMETER_VALUE"


def test_get_staging_timeout_defaults_to_two_hours(monkeypatch):
    monkeypatch.delenv("MLFLOW_DATABRICKS_VOLUME_WEIGHTS_STAGING_TIMEOUT", raising=False)
    assert get_staging_timeout() == 7200


@pytest.mark.parametrize("value", ["2h", "", "0", "-5"])
def test_get_staging_timeout_rejects_a_value_that_isnt_a_positive_number(monkeypatch, value):
    monkeypatch.setenv("MLFLOW_DATABRICKS_VOLUME_WEIGHTS_STAGING_TIMEOUT", value)
    with pytest.raises(MlflowException, match="must be a positive whole number of seconds") as e:
        get_staging_timeout()
    assert e.value.error_code == "INVALID_PARAMETER_VALUE"


def test_start_model_serving_staging_returns_the_status_polled_after_staging(stage, poll):
    poll.return_value = _status("IN_PROGRESS")
    host_creds = object()

    assert start_model_serving_staging("c.s.m", "3", host_creds) == _status("IN_PROGRESS")
    stage.assert_called_once_with("c.s.m", "3", host_creds=host_creds)
    poll.assert_called_once_with("c.s.m", "3", host_creds=host_creds)


def test_start_model_serving_staging_raises_with_the_server_reason_and_code(stage):
    stage.side_effect = RestException({
        "error_code": "BAD_REQUEST",
        "message": "it wasn't registered from a logged model.",
    })
    with pytest.raises(MlflowException, match="wasn't registered from a logged model") as e:
        start_model_serving_staging("c.s.m", "3", None)
    assert e.value.error_code == "BAD_REQUEST"


def test_start_model_serving_staging_explains_a_workspace_that_cant_stage_pending_versions(
    stage, poll
):
    stage.side_effect = RestException({
        "error_code": "NOT_FOUND",
        "message": "Model must be ready for it to be staged for serving.",
    })
    with pytest.raises(MlflowException, match="workspace can't stage model") as e:
        start_model_serving_staging("c.s.m", "3", None)
    assert e.value.error_code == "FEATURE_DISABLED"
    assert "Model must be ready" not in e.value.message
    poll.assert_not_called()


def test_start_model_serving_staging_maps_an_unknown_server_code_to_internal_error(stage):
    stage.side_effect = RestException({"error_code": "NOT_A_CLIENT_CODE", "message": "nope"})
    with pytest.raises(MlflowException, match="nope") as e:
        start_model_serving_staging("c.s.m", "3", None)
    assert e.value.error_code == "INTERNAL_ERROR"


@pytest.mark.parametrize(
    "poll_outcome",
    [{}, RestException({"error_code": "FEATURE_DISABLED", "message": "not enabled"})],
)
def test_start_model_serving_staging_raises_when_the_workspace_reports_no_status(
    stage, poll, poll_outcome
):
    if isinstance(poll_outcome, Exception):
        poll.side_effect = poll_outcome
    else:
        poll.return_value = poll_outcome
    with pytest.raises(MlflowException, match="doesn't report when Databricks finishes staging"):
        start_model_serving_staging("c.s.m", "3", None)


def test_start_model_serving_staging_propagates_other_status_errors(stage, poll):
    poll.side_effect = RestException({"error_code": "PERMISSION_DENIED", "message": "denied"})
    with pytest.raises(MlflowException, match="denied") as e:
        start_model_serving_staging("c.s.m", "3", None)
    assert e.value.error_code == "PERMISSION_DENIED"


def test_wait_returns_once_the_model_weights_are_copied(stage, poll, no_sleep):
    poll.side_effect = [_status("IN_PROGRESS"), _status("SUCCEEDED")]

    wait_for_model_artifact_staging("c.s.m", "3", _status("IN_PROGRESS"), None, 7200)

    assert poll.call_count == 2
    assert no_sleep.call_count == 2
    stage.assert_not_called()


def test_wait_ignores_the_environment_artifact(stage, poll):
    status = {
        "state": "IN_PROGRESS",
        "model_artifact": {"state": "SUCCEEDED"},
        "environment_artifact": {"state": "IN_PROGRESS"},
    }
    wait_for_model_artifact_staging("c.s.m", "3", status, None, 7200)
    poll.assert_not_called()


@pytest.mark.parametrize("failed_state", ["FAILED", "NOT_STAGED"])
def test_wait_restarts_failed_staging(stage, poll, failed_state):
    poll.side_effect = [_status("IN_PROGRESS"), _status("SUCCEEDED")]

    wait_for_model_artifact_staging("c.s.m", "3", _status(failed_state), None, 7200)

    stage.assert_called_once_with("c.s.m", "3", host_creds=None)


def test_wait_raises_once_restarts_are_exhausted(stage, poll):
    poll.return_value = _status("FAILED")

    with pytest.raises(
        MlflowException, match="while staging it for Model Serving. Check that the volume"
    ):
        wait_for_model_artifact_staging("c.s.m", "3", _status("FAILED"), None, 7200)

    assert stage.call_count == serving_staging._MAX_STAGING_RESTARTS
    # Each restart reads the status once while staging and once more after waiting.
    assert poll.call_count == 2 * serving_staging._MAX_STAGING_RESTARTS


def test_wait_raises_when_staging_outlasts_the_timeout(stage, poll):
    poll.return_value = _status("IN_PROGRESS")
    with (
        mock.patch.object(serving_staging.time, "monotonic", side_effect=[0, 10, 20, 31]),
        pytest.raises(MlflowException, match="within 30 seconds"),
    ):
        wait_for_model_artifact_staging("c.s.m", "3", _status("IN_PROGRESS"), None, 30)

    assert poll.call_count == 2
    stage.assert_not_called()


def test_wait_reports_failed_staging_without_restarting_it_after_the_timeout(stage, poll):
    with (
        mock.patch.object(serving_staging.time, "monotonic", side_effect=[0, 31]),
        pytest.raises(
            MlflowException, match="while staging it for Model Serving. Check that the volume"
        ),
    ):
        wait_for_model_artifact_staging("c.s.m", "3", _status("FAILED"), None, 30)

    stage.assert_not_called()
    poll.assert_not_called()


def test_wait_does_not_sleep_past_the_timeout(stage, poll, no_sleep):
    poll.return_value = _status("IN_PROGRESS")
    with (
        mock.patch.object(serving_staging.time, "monotonic", side_effect=[0, 10, 15]),
        pytest.raises(MlflowException, match="within 15 seconds"),
    ):
        wait_for_model_artifact_staging("c.s.m", "3", _status("IN_PROGRESS"), None, 15)

    no_sleep.assert_called_once_with(5)
    assert poll.call_count == 1


def test_wait_judges_a_restart_by_the_next_poll(stage, poll, no_sleep):
    # Staging reads FAILED right after the restart, before the server has re-armed it.
    poll.side_effect = [_status("FAILED"), _status("SUCCEEDED")]

    wait_for_model_artifact_staging("c.s.m", "3", _status("FAILED"), None, 7200)

    stage.assert_called_once_with("c.s.m", "3", host_creds=None)
    assert no_sleep.call_count == 1


def test_wait_keeps_polling_when_the_status_has_no_model_artifact(stage, poll):
    poll.return_value = _status("SUCCEEDED")

    wait_for_model_artifact_staging("c.s.m", "3", {"model_artifact": None}, None, 7200)

    poll.assert_called_once_with("c.s.m", "3", host_creds=None)
