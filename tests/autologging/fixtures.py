import os

import pytest

from mlflow.utils.autologging_utils import _is_testing, _AUTOLOGGING_TEST_MODE_ENV_VAR


@pytest.fixture
def test_mode_off():
    try:
        prev_env_var_value = os.environ.pop(_AUTOLOGGING_TEST_MODE_ENV_VAR, None)
        os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR] = "false"
        assert not _is_testing()
        yield
    finally:
        if prev_env_var_value:
            os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR] = prev_env_var_value
        else:
            del os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR]


@pytest.fixture
def test_mode_on():
    try:
        prev_env_var_value = os.environ.pop(_AUTOLOGGING_TEST_MODE_ENV_VAR, None)
        os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR] = "true"
        assert _is_testing()
        yield
    finally:
        if prev_env_var_value:
            os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR] = prev_env_var_value
        else:
            del os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR]
