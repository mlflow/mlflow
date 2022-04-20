import mlflow
from unittest import mock
from mlflow.utils.virtualenv import _create_virtualenv


from tests.projects.utils import TEST_VIRTUALENV_PROJECT_DIR, TEST_VIRTUALENV_CONDA_PROJECT_DIR


spy_on_create_virtualenv = mock.patch(
    "mlflow.projects.backend.local._create_virtualenv", wraps=_create_virtualenv
)


@spy_on_create_virtualenv
def test_virtualenv_project_execution(create_virtualenv_spy):
    submitted_run = mlflow.projects.run(TEST_VIRTUALENV_PROJECT_DIR, env_manager="virtualenv")
    submitted_run.wait()
    create_virtualenv_spy.assert_called_once()


@spy_on_create_virtualenv
def test_virtualenv_project_execution_without_env_manager(create_virtualenv_spy):
    # python_env project should be executed using virtualenv without explicitly specifying
    # env_manager="virtualenv"
    submitted_run = mlflow.projects.run(TEST_VIRTUALENV_PROJECT_DIR)
    submitted_run.wait()
    create_virtualenv_spy.assert_called_once()


@spy_on_create_virtualenv
def test_virtualenv_project_execution_local(create_virtualenv_spy):
    submitted_run = mlflow.projects.run(TEST_VIRTUALENV_PROJECT_DIR, env_manager="local")
    submitted_run.wait()
    create_virtualenv_spy.assert_not_called()


@spy_on_create_virtualenv
def test_virtualenv_conda_project_execution(create_virtualenv_spy):
    submitted_run = mlflow.projects.run(TEST_VIRTUALENV_CONDA_PROJECT_DIR, env_manager="virtualenv")
    submitted_run.wait()
    create_virtualenv_spy.assert_called_once()
