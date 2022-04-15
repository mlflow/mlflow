import pytest
from unittest import mock

from mlflow.entities import SourceType
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE
from mlflow.tracking.context.default_context import DefaultRunContext

# pylint: disable=unused-argument


MOCK_SCRIPT_NAME = "/path/to/script.py"


@pytest.fixture
def patch_script_name():
    patch_sys_argv = mock.patch("sys.argv", [MOCK_SCRIPT_NAME])
    patch_os_path_isfile = mock.patch("os.path.isfile", return_value=False)
    with patch_sys_argv, patch_os_path_isfile:
        yield


def test_default_run_context_in_context():
    assert DefaultRunContext().in_context() is True


def test_default_run_context_tags(patch_script_name):
    mock_user = mock.Mock()
    with mock.patch("getpass.getuser", return_value=mock_user):
        assert DefaultRunContext().tags() == {
            MLFLOW_USER: mock_user,
            MLFLOW_SOURCE_NAME: MOCK_SCRIPT_NAME,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.LOCAL),
        }
