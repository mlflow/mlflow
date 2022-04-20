import mlflow

from tests.projects.utils import TEST_VIRTUALENV_PROJECT_DIR, TEST_VIRTUALENV_CONDA_PROJECT_DIR


def test_virtualenv_project_execution():
    submitted_run = mlflow.projects.run(TEST_VIRTUALENV_PROJECT_DIR, env_manager="virtualenv")
    submitted_run.wait()


def test_virtualenv_project_execution_without_env_manager():
    # Project should be executed using virtualenv without explicitly specifing `env_manager`
    submitted_run = mlflow.projects.run(TEST_VIRTUALENV_PROJECT_DIR)
    submitted_run.wait()


def test_virtualenv_conda_project_execution():
    submitted_run = mlflow.projects.run(TEST_VIRTUALENV_CONDA_PROJECT_DIR, env_manager="virtualenv")
    submitted_run.wait()
