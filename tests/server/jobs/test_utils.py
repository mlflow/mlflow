import os

import pytest

from mlflow.exceptions import MlflowException
from mlflow.server.jobs.utils import _validate_function_parameters

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="MLflow job execution is not supported on Windows"
)


def test_validate_function_parameters():
    def test_func(a, b, c=None):
        return a + b + (c or 0)

    # Test with all required parameters present
    _validate_function_parameters(test_func, {"a": 1, "b": 2})
    _validate_function_parameters(test_func, {"a": 1, "b": 2, "c": 3})

    # Test with missing required parameters
    with pytest.raises(MlflowException, match=r"Missing required parameters.*\['b'\]"):
        _validate_function_parameters(test_func, {"a": 1})

    # Test with multiple missing required parameters
    with pytest.raises(MlflowException, match=r"Missing required parameters.*\['a', 'b'\]"):
        _validate_function_parameters(test_func, {})


def test_validate_function_parameters_with_varargs():
    def test_func_with_kwargs(a, **kwargs):
        return a

    # Should not raise error even with extra parameters due to **kwargs
    _validate_function_parameters(test_func_with_kwargs, {"a": 1, "extra": 2})

    # Should still raise error for missing required parameters
    with pytest.raises(MlflowException, match=r"Missing required parameters.*\['a'\]"):
        _validate_function_parameters(test_func_with_kwargs, {"extra": 2})


def test_validate_function_parameters_with_positional_args():
    def test_func_with_args(a, *args):
        return a

    # Should work fine with just required parameter
    _validate_function_parameters(test_func_with_args, {"a": 1})

    # Should still raise error for missing required parameters
    with pytest.raises(MlflowException, match=r"Missing required parameters.*\['a'\]"):
        _validate_function_parameters(test_func_with_args, {})


def test_job_status_conversion():
    from mlflow.entities._job_status import JobStatus

    assert JobStatus.from_int(1) == JobStatus.RUNNING
    assert JobStatus.from_str("RUNNING") == JobStatus.RUNNING

    assert JobStatus.RUNNING.to_int() == 1
    assert str(JobStatus.RUNNING) == "RUNNING"

    with pytest.raises(
        MlflowException, match="The value -1 can't be converted to JobStatus enum value."
    ):
        JobStatus.from_int(-1)

    with pytest.raises(
        MlflowException, match="The value 5 can't be converted to JobStatus enum value."
    ):
        JobStatus.from_int(5)

    with pytest.raises(
        MlflowException, match="The string 'ABC' can't be converted to JobStatus enum value."
    ):
        JobStatus.from_str("ABC")
