import pytest
from unittest import mock

import mlflow.utils.autologging_utils as autologging_utils


@pytest.fixture
def test_mode_off():
    with mock.patch("mlflow.utils.autologging_utils._is_testing") as testing_mock:
        testing_mock.return_value = False
        assert not autologging_utils._is_testing()
        yield


@pytest.fixture
def test_mode_on():
    with mock.patch("mlflow.utils.autologging_utils._is_testing") as testing_mock:
        testing_mock.return_value = True
        assert autologging_utils._is_testing()
        yield
