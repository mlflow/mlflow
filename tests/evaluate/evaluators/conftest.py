import numpy as np
import pytest


@pytest.fixture(autouse=True)
def suppress_dummy_evaluator():
    """
    Dummy evaluator is registered by the test plugin and used in
    test_evaluation.py, but we don't want it to be used in this test.

    This fixture suppress dummy evaluator for the duration of each test.
    """
    from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry

    dummy_evaluator = _model_evaluation_registry._registry.pop("dummy_evaluator")

    yield

    _model_evaluation_registry._registry["dummy_evaluator"] = dummy_evaluator


def assert_dict_equal(d1, d2, rtol):
    for k in d1:
        assert k in d2
        assert np.isclose(d1[k], d2[k], rtol=rtol)


def assert_metrics_equal(actual, expected):
    for metric_key in expected:
        assert np.isclose(expected[metric_key], actual[metric_key], rtol=1e-3)
