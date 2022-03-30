import pytest

from mlflow.utils.conda import get_or_create_conda_env
from mlflow.utils.process import ShellCommandException


def test_get_or_create_conda_env_capture_output_mode(tmp_path):
    conda_yaml_file = tmp_path / "conda.yaml"
    conda_yaml_file.write_text(
        """
channels:
- conda-forge
dependencies:
- pip:
  - scikit-learn==99.99.99
"""
    )
    with pytest.raises(
        ShellCommandException,
        match="Could not find a version that satisfies the requirement scikit-learn==99.99.99",
    ):
        get_or_create_conda_env(str(conda_yaml_file), capture_output=True)
