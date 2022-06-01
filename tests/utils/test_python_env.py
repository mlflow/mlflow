import pytest
from unittest import mock

from mlflow.utils.environment import _PythonEnv


def test_constructor_argument_validation():
    with pytest.raises(TypeError, match="`python` must be a string"):
        _PythonEnv(python=1)

    with pytest.raises(TypeError, match="`build_dependencies` must be a list"):
        _PythonEnv(build_dependencies=0)

    with pytest.raises(TypeError, match="`dependencies` must be a list"):
        _PythonEnv(dependencies=0)


def test_to_yaml(tmp_path):
    yaml_path = tmp_path / "python_env.yaml"
    _PythonEnv("3.7.5", ["a"], ["b"]).to_yaml(yaml_path)
    expected_content = """
python: 3.7.5
build_dependencies:
- a
dependencies:
- b
""".lstrip()
    assert yaml_path.read_text() == expected_content


def test_from_yaml(tmp_path):
    content = """
python: 3.7.5
build_dependencies:
- a
- b
dependencies:
- c
- d
"""
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(content)
    python_env = _PythonEnv.from_yaml(yaml_path)
    assert python_env.python == "3.7.5"
    assert python_env.build_dependencies == ["a", "b"]
    assert python_env.dependencies == ["c", "d"]


def test_from_conda_yaml(tmp_path):
    content = """
name: example
channels:
  - conda-forge
dependencies:
  - python=3.7.5
  - pip
  - pip:
    - a
    - b
"""
    yaml_path = tmp_path / "conda.yaml"
    yaml_path.write_text(content)
    python_env = _PythonEnv.from_conda_yaml(yaml_path)
    assert python_env.python == "3.7.5"
    assert python_env.build_dependencies == ["pip"]
    assert python_env.dependencies == ["a", "b"]


def test_from_conda_yaml_build_dependencies(tmp_path):
    content = """
name: example
channels:
  - conda-forge
dependencies:
  - python=3.7.5
  - pip=1.2.3
  - wheel==4.5.6
  - setuptools<=7.8.9
  - pip:
    - a
    - b
"""
    yaml_path = tmp_path / "conda.yaml"
    yaml_path.write_text(content)
    python_env = _PythonEnv.from_conda_yaml(yaml_path)
    assert python_env.python == "3.7.5"
    assert python_env.build_dependencies == ["pip==1.2.3", "wheel==4.5.6", "setuptools<=7.8.9"]
    assert python_env.dependencies == ["a", "b"]


def test_from_conda_yaml_python_dependency_is_missing(tmp_path):
    content = """
name: example
channels:
  - conda-forge
dependencies:
  - pip
  - pip:
    - a
    - b
"""
    yaml_path = tmp_path / "conda.yaml"
    yaml_path.write_text(content)
    with pytest.raises(Exception, match="Could not extract python version"):
        _PythonEnv.from_conda_yaml(yaml_path)


def test_from_conda_yaml_invalid_python_comparator(tmp_path):
    content = """
name: example
channels:
  - conda-forge
dependencies:
  - python<3.7.9
  - pip:
    - a
    - b
"""
    yaml_path = tmp_path / "conda.yaml"
    yaml_path.write_text(content)
    with pytest.raises(Exception, match="Invalid version comparator for python"):
        _PythonEnv.from_conda_yaml(yaml_path)


def test_from_conda_yaml_conda_dependencies_warning(tmp_path):
    content = """
name: example
channels:
  - conda-forge
dependencies:
  - python=3.8
  - foo
  - bar
  - pip:
    - a
"""
    yaml_path = tmp_path / "conda.yaml"
    yaml_path.write_text(content)
    with mock.patch("mlflow.utils.environment._logger.warning") as mock_warning:
        _PythonEnv.from_conda_yaml(yaml_path)
        mock_warning.assert_called_with(
            "The following conda dependencies will not be installed "
            "in the resulting environment: %s",
            ["foo", "bar"],
        )
