import pytest

from mlflow.utils.environment import PythonEnv


def test_constructor_argument_validation():
    with pytest.raises(TypeError, match="`python` must be a string"):
        PythonEnv(python=1)

    with pytest.raises(TypeError, match="`build_dependencies` must be a list"):
        PythonEnv(build_dependencies=0)

    with pytest.raises(TypeError, match="`dependencies` must be a list"):
        PythonEnv(dependencies=0)


def test_to_yaml(tmp_path):
    yaml_path = tmp_path / "python-env.yaml"
    PythonEnv("3.7.5", ["a"], ["b"]).to_yaml(yaml_path)
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
    python_env = PythonEnv.from_yaml(yaml_path)
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
    python_env = PythonEnv.from_conda_yaml(yaml_path)
    assert python_env.python == "3.7.5"
    assert python_env.build_dependencies is None
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
    python_env = PythonEnv.from_conda_yaml(yaml_path)
    assert python_env.python == "3.7.5"
    assert python_env.build_dependencies == ["pip==1.2.3", "wheel==4.5.6", "setuptools<=7.8.9"]
    assert python_env.dependencies == ["a", "b"]


def test_from_conda_yaml_missing_python(tmp_path):
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
    with pytest.raises(Exception, match="Failed to create a `PythonEnv` object"):
        PythonEnv.from_conda_yaml(yaml_path)
