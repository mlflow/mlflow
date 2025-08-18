import threading
from unittest import mock

import pytest

import mlflow
from mlflow.environment_variables import MLFLOW_TRACKING_USERNAME
from mlflow.genai.evaluation.context import NoneContext, _set_context, eval_context, get_context


@pytest.fixture(autouse=True)
def reset_context():
    yield
    _set_context(NoneContext())


def test_context_get_experiment_and_run_id():
    exp_id = mlflow.set_experiment("Test").experiment_id

    @eval_context
    def _test():
        assert exp_id == get_context().get_mlflow_experiment_id()
        assert get_context().get_mlflow_run_id() is None

    _test()


def test_context_get_run_id_active_run():
    @eval_context
    def _test():
        with mlflow.start_run() as run:
            assert run.info.run_id == get_context().get_mlflow_run_id()

    _test()


def test_context_get_run_id_explicitly_set():
    @eval_context
    def _test():
        context = get_context()
        context.set_mlflow_run_id("test-run-id")
        assert context.get_mlflow_run_id() == "test-run-id"

        run_id = None

        def _target():
            nonlocal run_id
            run_id = get_context().get_mlflow_run_id()

        thread = threading.Thread(target=_target)
        thread.start()
        assert run_id == "test-run-id"

    _test()


def test_context_get_user_name(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "test-user")

    @eval_context
    def _test():
        assert get_context().get_user_name() == "test-user"

    _test()


@mock.patch("mlflow.tracking.context.default_context.DefaultRunContext.tags", return_value={})
def test_context_get_user_name_no_user_set(mock_default_tags):
    @eval_context
    def _test():
        assert get_context().get_user_name() == "unknown"

    _test()
