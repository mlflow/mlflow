import pytest

import mlflow


@pytest.fixture
def reset_active_experiment():
    yield
    mlflow.tracking.fluent._active_experiment_id = None
