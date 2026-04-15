import os

import pytest

from mlflow.exceptions import MlflowException
from mlflow.server.jobs.utils import _load_function, _validate_function_parameters

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
        MlflowException, match="The value 6 can't be converted to JobStatus enum value."
    ):
        JobStatus.from_int(6)

    with pytest.raises(
        MlflowException, match="The string 'ABC' can't be converted to JobStatus enum value."
    ):
        JobStatus.from_str("ABC")


def test_load_function_invalid_function_format():
    with pytest.raises(MlflowException, match="Invalid function fullname format"):
        _load_function("invalid_format_no_module")


def test_load_function_module_not_found():
    with pytest.raises(MlflowException, match="Module not found"):
        _load_function("non_existent_module.some_function")


def test_load_function_function_not_found():
    with pytest.raises(MlflowException, match="Function not found in module"):
        _load_function("os.non_exist_function")


def test_compute_exclusive_lock_key():
    from mlflow.server.jobs.utils import _compute_exclusive_lock_key

    # Same params produce same key
    key1 = _compute_exclusive_lock_key("job_name", {"a": 1, "b": 2})
    key2 = _compute_exclusive_lock_key("job_name", {"a": 1, "b": 2})
    assert key1 == key2

    # Order doesn't matter for params
    key3 = _compute_exclusive_lock_key("job_name", {"b": 2, "a": 1})
    assert key1 == key3

    # Different params produce different keys
    key4 = _compute_exclusive_lock_key("job_name", {"a": 1, "b": 3})
    assert key1 != key4

    # Different job names produce different keys
    key5 = _compute_exclusive_lock_key("other_job", {"a": 1, "b": 2})
    assert key1 != key5

    # Test with filtered params (simulating exclusive parameter list)
    # When only "a" is used, different "b" values should produce same key
    key6 = _compute_exclusive_lock_key("job_name", {"a": 1})
    key7 = _compute_exclusive_lock_key("job_name", {"a": 1})
    assert key6 == key7

    # But different "a" values should produce different keys
    key8 = _compute_exclusive_lock_key("job_name", {"a": 2})
    assert key6 != key8

    # Test that same filtered params produce same key
    filtered_params = {"a": 1, "b": 2}
    key9 = _compute_exclusive_lock_key("job_name", filtered_params)
    key10 = _compute_exclusive_lock_key("job_name", {"a": 1, "b": 2})
    assert key9 == key10

    # Different filtered params produce different keys
    key11 = _compute_exclusive_lock_key("job_name", {"a": 1, "b": 3})
    assert key9 != key11

    # Key format is job_name:hash
    assert key1.startswith("job_name:")
    assert key5.startswith("other_job:")
