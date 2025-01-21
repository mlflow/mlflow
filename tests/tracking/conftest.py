import pytest

import mlflow


@pytest.fixture
def reset_active_experiment():
    yield
    mlflow.tracking.fluent._active_experiment_id = None

def pytest_generate_tests(metafunc):
    if "synchronous" in metafunc.fixturenames:
        metafunc.parametrize("synchronous", [True, False])
