import pytest

import mlflow
from mlflow.utils.async_logging.run_operations import RunOperations


@pytest.fixture
def reset_active_experiment():
    yield
    mlflow.tracking.fluent._active_experiment_id = None


def pytest_generate_tests(metafunc):
    if "synchronous" in metafunc.fixturenames:
        metafunc.parametrize("synchronous", [True, False])


def _log_figure_with_sync(interface, synchronous, *args, **kwargs):
    task = interface.log_figure(*args, **kwargs, synchronous=synchronous)
    if isinstance(task, RunOperations):
        task.wait(30)


def _log_image_with_sync(interface, synchronous, *args, **kwargs):
    task = interface.log_image(*args, **kwargs, synchronous=synchronous)
    if isinstance(task, RunOperations):
        task.wait(timeout=30)


def _log_artifact_with_sync(interface, synchronous, *args, **kwargs):
    task = interface.log_artifact(*args, **kwargs, synchronous=synchronous)
    if isinstance(task, RunOperations):
        task.wait(timeout=30)


def _log_artifacts_with_sync(interface, synchronous, *args, **kwargs):
    task = interface.log_artifacts(*args, **kwargs, synchronous=synchronous)
    if isinstance(task, RunOperations):
        task.wait(timeout=30)


def _log_text_with_sync(interface, synchronous, *args, **kwargs):
    task = interface.log_text(*args, **kwargs, synchronous=synchronous)
    if isinstance(task, RunOperations):
        task.wait(timeout=30)


def _log_table_with_sync(interface, synchronous, *args, **kwargs):
    task = interface.log_table(*args, **kwargs, synchronous=synchronous)
    if isinstance(task, RunOperations):
        task.wait(timeout=30)
