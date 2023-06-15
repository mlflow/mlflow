import os

import pytest

import mlflow


@pytest.fixture
def tmp_wkdir(tmp_path):
    initial_wkdir = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(initial_wkdir)


@pytest.fixture
def reset_active_experiment():
    yield
    mlflow.tracking.fluent._active_experiment_id = None
