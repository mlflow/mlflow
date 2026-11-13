from types import SimpleNamespace
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException, RestException
from mlflow.store._unity_catalog.registry import serving_staging
from mlflow.store._unity_catalog.registry.serving_staging import (
    check_volume_staging_source,
    get_staging_timeout,
    get_volume_artifacts_path,
    is_model_version_copy,
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


def test_start_model_serving_staging_explains_why_a_pending_version_wasnt_staged(stage, poll):
    stage.side_effect = RestException({
        "error_code": "NOT_FOUND",
        "message": "Model must be ready for it to be staged for serving.",
    })
    with pytest.raises(MlflowException, match="couldn't stage model 'c.s.m' version 3") as e:
        start_model_serving_staging("c.s.m", "3", None)
    assert e.value.error_code == "NOT_FOUND"
    # Every reason the server may have had is named, and retrying comes first.
    assert "may not support this yet" in e.value.message
    assert "may not be eligible" in e.value.message
    assert "failed temporarily. Try registering the model again." in e.value.message
    assert "without 'volume_artifacts_path' in its metadata" in e.value.message
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


def test_wait_ignores_the_environment_artifact(stage, poll, caplog):
    status = {
        "state": "IN_PROGRESS",
        "model_artifact": {"state": "SUCCEEDED"},
        "environment_artifact": {"state": "IN_PROGRESS"},
    }
    with caplog.at_level("WARNING", logger=_MODULE):
        wait_for_model_artifact_staging("c.s.m", "3", status, None, 7200)
    poll.assert_not_called()
    assert "serving environment" not in caplog.text


def test_wait_warns_when_the_environment_already_failed(stage, poll, caplog):
    status = {
        "state": "FAILED",
        "model_artifact": {"state": "SUCCEEDED"},
        "environment_artifact": {"state": "FAILED"},
    }
    with caplog.at_level("WARNING", logger=_MODULE):
        wait_for_model_artifact_staging("c.s.m", "3", status, None, 7200)
    # The version is still finalized: its files are complete.
    poll.assert_not_called()
    stage.assert_not_called()
    assert "couldn't prepare the serving environment of model 'c.s.m' version 3" in caplog.text


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


def test_wait_does_not_sleep_past_the_timeout_after_a_restart(stage, poll, no_sleep):
    # The restart itself uses up the rest of the 30 seconds.
    poll.return_value = _status("IN_PROGRESS")
    with (
        mock.patch.object(serving_staging.time, "monotonic", side_effect=[0, 10, 35, 36]),
        pytest.raises(MlflowException, match="within 30 seconds"),
    ):
        wait_for_model_artifact_staging("c.s.m", "3", _status("FAILED"), None, 30)

    stage.assert_called_once_with("c.s.m", "3", host_creds=None)
    no_sleep.assert_called_once_with(0)


def test_wait_reports_a_restart_that_failed_again_after_the_timeout(stage, poll, no_sleep):
    poll.side_effect = [_status("IN_PROGRESS"), _status("FAILED")]
    with (
        mock.patch.object(serving_staging.time, "monotonic", side_effect=[0, 10, 35, 36]),
        pytest.raises(MlflowException, match="Check that the volume path"),
    ):
        wait_for_model_artifact_staging("c.s.m", "3", _status("FAILED"), None, 30)

    stage.assert_called_once()
    no_sleep.assert_called_once_with(0)


_VOLUME_PATH = "/Volumes/main/default/weights/qwen"


@pytest.mark.parametrize(
    "source",
    ["models:/c.s.m/3", "models:/c.s.m@champion", "models:/my model/3", "models:/a@b/12"],
)
def test_is_model_version_copy_recognizes_registered_model_versions(source):
    assert is_model_version_copy(source)


@pytest.mark.parametrize(
    "source",
    [
        "models:/m-0123456789abcdef0123456789abcdef",
        "models:/m-0123456789abcdef0123456789abcdef/artifacts",
        "models:/c.s.m",
        "models:/",
        "models:/c.s.m/3/extra",
        "models:/c.s.m/3?x=1",
        "models:/c.s.m/3#x",
        "models:/c.s.m/3 ",
        " models:/c.s.m/3",
        "models:/c.s.m/latest",
        "models:/c.s.m/Production",
        "models:/c.s.m@",
        "models://[x",
        "runs:/abc/model",
        "/local/model",
        "dbfs:/databricks/mlflow-tracking/1/abc/artifacts/model",
        None,
    ],
)
def test_is_model_version_copy_rejects_other_sources(source):
    assert not is_model_version_copy(source)


@pytest.mark.parametrize(
    "source",
    [
        "models:/m-0123456789abcdef0123456789abcdef",
        "dbfs:/databricks/mlflow-tracking/1/logged_models/m-0123456789abcdef/artifacts",
        "s3://bucket/1/logged_models/m-0123456789abcdef/artifacts/",
    ],
)
def test_check_volume_staging_source_accepts_logged_models(source, tmp_path):
    check_volume_staging_source(source, _VOLUME_PATH, tmp_path)


@pytest.mark.parametrize("volume_path", [_VOLUME_PATH, _VOLUME_PATH + "/"])
def test_check_volume_staging_source_accepts_a_copy_that_holds_the_weights(volume_path, tmp_path):
    tmp_path.joinpath("artifacts", "qwen").mkdir(parents=True)
    tmp_path.joinpath("artifacts", "qwen", "model.safetensors").write_text("weights")
    check_volume_staging_source("models:/c.s.m/3", volume_path, tmp_path)


@pytest.mark.parametrize("weights", [None, "empty", "file", "symlink"])
def test_check_volume_staging_source_rejects_a_copy_without_the_weights(weights, tmp_path):
    # A version copied from a registry that doesn't stage weights, or registered before staging.
    tmp_path.joinpath("artifacts").mkdir()
    weights_dir = tmp_path.joinpath("artifacts", "qwen")
    if weights == "empty":
        weights_dir.mkdir()
    elif weights == "file":
        weights_dir.write_text("not a directory")
    elif weights == "symlink":
        # The upload doesn't follow a symlinked directory.
        elsewhere = tmp_path.joinpath("elsewhere")
        elsewhere.mkdir()
        elsewhere.joinpath("model.safetensors").write_text("weights")
        weights_dir.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(MlflowException, match="doesn't contain the weights") as e:
        check_volume_staging_source("models:/c.s.m/3", _VOLUME_PATH, tmp_path)
    assert e.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "('models:/<model_id>')" in e.value.message
    assert "without 'volume_artifacts_path' in its metadata" in e.value.message


def test_check_volume_staging_source_names_the_logged_model_to_register(tmp_path):
    model_id = "m-0123456789abcdef0123456789abcdef"
    with pytest.raises(MlflowException, match=f"'models:/{model_id}'"):
        check_volume_staging_source("models:/c.s.m/3", _VOLUME_PATH, tmp_path, model_id=model_id)
    with pytest.raises(
        MlflowException, match=f"Register it by its logged model URI 'models:/{model_id}'"
    ):
        check_volume_staging_source("runs:/abc/model", _VOLUME_PATH, tmp_path, model_id=model_id)


@pytest.mark.parametrize(
    "source",
    [
        "runs:/abc/model",
        "/local/model",
        "dbfs:/databricks/mlflow-tracking/1/abc/artifacts/model",
        "dbfs:/databricks/mlflow-tracking/1/logged_models/m-0123456789abcdef/artifacts/model",
        "models:/c.s.m",
        "models:/c.s.m/latest",
        "models:/M-0123456789ABCDEF",
        "",
    ],
)
def test_check_volume_staging_source_rejects_sources_without_a_logged_model(source, tmp_path):
    with pytest.raises(MlflowException, match="must be registered from its logged model") as e:
        check_volume_staging_source(source, _VOLUME_PATH, tmp_path)
    assert e.value.error_code == "INVALID_PARAMETER_VALUE"


def test_wait_keeps_polling_when_the_status_has_no_model_artifact(stage, poll):
    poll.return_value = _status("SUCCEEDED")

    wait_for_model_artifact_staging("c.s.m", "3", {"model_artifact": None}, None, 7200)

    poll.assert_called_once_with("c.s.m", "3", host_creds=None)
