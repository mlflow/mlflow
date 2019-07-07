import mock

from mlflow.utils.conda_utils import MLFLOW_CONDA_HOME, _get_conda_bin_executable


def test_conda_path():
    """Verify that we correctly determine the path to conda executables"""
    with mock.patch.dict("os.environ", {MLFLOW_CONDA_HOME: "/some/dir/"}):
        assert _get_conda_bin_executable("conda") == "/some/dir/bin/conda"
        assert _get_conda_bin_executable("activate") == "/some/dir/bin/activate"
    with mock.patch.dict("os.environ", {}):
        assert _get_conda_bin_executable("conda") == "conda"
