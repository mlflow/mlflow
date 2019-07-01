import mock
import pytest

from mlflow.utils.conda_utils import MLFLOW_CONDA_HOME, _get_conda_bin_executable


@pytest.mark.parametrize(
    "mock_env,expected_conda,expected_activate",
    [
        ({}, "conda", "activate"),
        ({MLFLOW_CONDA_HOME: "/some/dir/"}, "/some/dir/bin/conda",
         "/some/dir/bin/activate")
    ]
)
def test_conda_path(mock_env, expected_conda, expected_activate):
    """Verify that we correctly determine the path to conda executables"""
    with mock.patch.dict("os.environ", mock_env):
        assert _get_conda_bin_executable("conda") == expected_conda
        assert _get_conda_bin_executable("activate") == expected_activate
