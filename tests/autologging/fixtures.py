import os
import sys

import pytest

import mlflow.utils.logging_utils as logging_utils
from mlflow.utils.autologging_utils import is_testing
from mlflow.utils.autologging_utils.safety import _AUTOLOGGING_TEST_MODE_ENV_VAR


PATCH_DESTINATION_FN_DEFAULT_RESULT = "original_result"


@pytest.fixture
def patch_destination():
    class PatchObj:
        def __init__(self):
            self.fn_call_count = 0
            self.recurse_fn_call_count = 0

        def fn(self, *args, **kwargs):  # pylint: disable=unused-argument
            self.fn_call_count += 1
            return PATCH_DESTINATION_FN_DEFAULT_RESULT

        def recursive_fn(self, level, max_depth):
            self.recurse_fn_call_count += 1
            if level == max_depth:
                return PATCH_DESTINATION_FN_DEFAULT_RESULT
            else:
                return self.recursive_fn(level + 1, max_depth)

        def throw_error_fn(self, error_to_raise):
            raise error_to_raise

    return PatchObj()


@pytest.fixture
def test_mode_off():
    try:
        prev_env_var_value = os.environ.pop(_AUTOLOGGING_TEST_MODE_ENV_VAR, None)
        os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR] = "false"
        assert not is_testing()
        yield
    finally:
        if prev_env_var_value:
            os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR] = prev_env_var_value
        else:
            del os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR]


def enable_test_mode():
    try:
        prev_env_var_value = os.environ.pop(_AUTOLOGGING_TEST_MODE_ENV_VAR, None)
        os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR] = "true"
        assert is_testing()
        yield
    finally:
        if prev_env_var_value:
            os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR] = prev_env_var_value
        else:
            del os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR]


@pytest.fixture
def test_mode_on():
    yield from enable_test_mode()


@pytest.fixture(autouse=True)
def reset_stderr():
    prev_stderr = sys.stderr
    yield
    sys.stderr = prev_stderr


@pytest.fixture(autouse=True)
def reset_logging_enablement():
    yield
    logging_utils.enable_logging()
