import mlflow

from tests.projects.utils import TEST_VIRTUALENV_PROJECT_DIR


def test_virtualenv_project_execution():
    submitted_run = mlflow.projects.run(TEST_VIRTUALENV_PROJECT_DIR, env_manager="virtualenv")
    submitted_run.wait()
