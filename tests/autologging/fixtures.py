import os
import sys

import pytest

from mlflow.environment_variables import _MLFLOW_AUTOLOGGING_TESTING
from mlflow.utils import logging_utils
from mlflow.utils.autologging_utils import is_testing

PATCH_DESTINATION_FN_DEFAULT_RESULT = "original_result"


# Fixture to run the test case with and without async logging enabled
@pytest.fixture(params=[True, False], ids=["sync", "async"])
def patch_destination(request):
    if request.param:

        class Destination:
            def __init__(self):
                self.fn_call_count = 0
                self.recurse_fn_call_count = 0

            def fn(self, *args, **kwargs):
                self.fn_call_count += 1
                return PATCH_DESTINATION_FN_DEFAULT_RESULT

            def fn2(self, *args, **kwargs):
                return "f2"

            def recursive_fn(self, level, max_depth):
                self.recurse_fn_call_count += 1
                if level == max_depth:
                    return PATCH_DESTINATION_FN_DEFAULT_RESULT
                else:
                    return self.recursive_fn(level + 1, max_depth)

            def throw_error_fn(self, error_to_raise):
                raise error_to_raise

            @property
            def is_async(self):
                return False

    else:

        class Destination:
            def __init__(self):
                self.fn_call_count = 0
                self.recurse_fn_call_count = 0

            async def fn(self, *args, **kwargs):
                self.fn_call_count += 1
                return PATCH_DESTINATION_FN_DEFAULT_RESULT

            async def fn2(self, *args, **kwargs):
                return "f2"

            async def recursive_fn(self, level, max_depth):
                self.recurse_fn_call_count += 1
                if level == max_depth:
                    return PATCH_DESTINATION_FN_DEFAULT_RESULT
                else:
                    return await self.recursive_fn(level + 1, max_depth)

            async def throw_error_fn(self, error_to_raise):
                raise error_to_raise

            @property
            def is_async(self):
                return True

    return Destination()


@pytest.fixture
def test_mode_off():
    prev_env_var_value = os.environ.pop(_MLFLOW_AUTOLOGGING_TESTING.name, None)
    try:
        os.environ[_MLFLOW_AUTOLOGGING_TESTING.name] = "false"
        assert not is_testing()
        yield
    finally:
        if prev_env_var_value:
            os.environ[_MLFLOW_AUTOLOGGING_TESTING.name] = prev_env_var_value
        else:
            del os.environ[_MLFLOW_AUTOLOGGING_TESTING.name]


def enable_test_mode():
    prev_env_var_value = os.environ.pop(_MLFLOW_AUTOLOGGING_TESTING.name, None)
    try:
        os.environ[_MLFLOW_AUTOLOGGING_TESTING.name] = "true"
        assert is_testing()
        yield
    finally:
        if prev_env_var_value:
            os.environ[_MLFLOW_AUTOLOGGING_TESTING.name] = prev_env_var_value
        else:
            del os.environ[_MLFLOW_AUTOLOGGING_TESTING.name]


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
