import os
import pytest

from mlflow.utils.environment import (
    _mlflow_conda_env,
    _parse_pip_requirements,
    _validate_env_arguments,
)


@pytest.fixture
def conda_env_path(tmpdir):
    return os.path.join(tmpdir.strpath, "conda_env.yaml")


def test_mlflow_conda_env_returns_none_when_output_path_is_specified(conda_env_path):
    env_creation_output = _mlflow_conda_env(
        path=conda_env_path,
        additional_conda_deps=["conda-dep-1=0.0.1", "conda-dep-2"],
        additional_pip_deps=["pip-dep-1", "pip-dep2==0.1.0"],
    )

    assert env_creation_output is None


def test_mlflow_conda_env_returns_expected_env_dict_when_output_path_is_not_specified():
    conda_deps = ["conda-dep-1=0.0.1", "conda-dep-2"]
    env = _mlflow_conda_env(path=None, additional_conda_deps=conda_deps)

    for conda_dep in conda_deps:
        assert conda_dep in env["dependencies"]


@pytest.mark.parametrize("conda_deps", [["conda-dep-1=0.0.1", "conda-dep-2"], None])
def test_mlflow_conda_env_includes_pip_dependencies_but_pip_is_not_specified(conda_deps):
    additional_pip_deps = ["pip-dep==0.0.1"]
    env = _mlflow_conda_env(
        path=None, additional_conda_deps=conda_deps, additional_pip_deps=additional_pip_deps
    )
    if conda_deps is not None:
        for conda_dep in conda_deps:
            assert conda_dep in env["dependencies"]
    assert "pip" in env["dependencies"]


@pytest.mark.parametrize("pip_specification", ["pip", "pip==20.0.02"])
def test_mlflow_conda_env_includes_pip_dependencies_and_pip_is_specified(pip_specification):
    conda_deps = ["conda-dep-1=0.0.1", "conda-dep-2", pip_specification]
    additional_pip_deps = ["pip-dep==0.0.1"]
    env = _mlflow_conda_env(
        path=None, additional_conda_deps=conda_deps, additional_pip_deps=additional_pip_deps
    )
    for conda_dep in conda_deps:
        assert conda_dep in env["dependencies"]
    assert pip_specification in env["dependencies"]
    assert env["dependencies"].count("pip") == (2 if pip_specification == "pip" else 1)


def test_parse_pip_requirements(tmpdir):
    assert _parse_pip_requirements(None) == []
    assert _parse_pip_requirements([]) == []
    # Without version specifiers
    assert _parse_pip_requirements(["a", "b"]) == ["a", "b"]
    # With version specifiers
    assert _parse_pip_requirements(["a==0.0", "b>1.1"]) == ["a==0.0", "b>1.1"]
    # Environment marker (https://www.python.org/dev/peps/pep-0508/#environment-markers)
    assert _parse_pip_requirements(['a; python_version < "3.8"']) == ['a; python_version < "3.8"']
    # GitHub URI
    mlflow_repo_uri = "git+https://github.com/mlflow/mlflow.git"
    assert _parse_pip_requirements([mlflow_repo_uri]) == [mlflow_repo_uri]
    # Local file
    fake_whl = tmpdir.join("fake.whl")
    fake_whl.write("")
    assert _parse_pip_requirements([fake_whl.strpath]) == [fake_whl.strpath]


def test_parse_pip_requirements_with_relative_requirements_files(request, tmpdir):
    try:
        os.chdir(tmpdir)
        f1 = tmpdir.join("requirements1.txt")
        f1.write("b")
        assert _parse_pip_requirements(f1.basename) == ["b"]
        assert _parse_pip_requirements(["a", f"-r {f1.basename}"]) == ["a", "b"]

        f2 = tmpdir.join("requirements2.txt")
        f3 = tmpdir.join("requirements3.txt")
        f2.write(f"b\n-r {f3.basename}")
        f3.write("c")
        assert _parse_pip_requirements(f2.basename) == ["b", "c"]
        assert _parse_pip_requirements(["a", f"-r {f2.basename}"]) == ["a", "b", "c"]
    finally:
        os.chdir(request.config.invocation_dir)


def test_parse_pip_requirements_with_absolute_requirements_files(tmpdir):
    f1 = tmpdir.join("requirements1.txt")
    f1.write("b")
    assert _parse_pip_requirements(f1.strpath) == ["b"]
    assert _parse_pip_requirements(["a", f"-r {f1.strpath}"]) == ["a", "b"]

    f2 = tmpdir.join("requirements2.txt")
    f3 = tmpdir.join("requirements3.txt")
    f2.write(f"b\n-r {f3.strpath}")
    f3.write("c")
    assert _parse_pip_requirements(f2.strpath) == ["b", "c"]
    assert _parse_pip_requirements(["a", f"-r {f2.strpath}"]) == ["a", "b", "c"]


def test_parse_pip_requirements_ignores_comments_and_blank_lines(tmpdir):
    reqs = [
        "# comment",
        "a # inline comment",
        # blank lines
        "",
        " ",
    ]
    f = tmpdir.join("requirements.txt")
    f.write("\n".join(reqs))
    assert _parse_pip_requirements(reqs) == ["a"]
    assert _parse_pip_requirements(f.strpath) == ["a"]


def test_parse_pip_requirements_removes_temporary_requirements_file():
    assert _parse_pip_requirements(["a"]) == ["a"]
    assert all(not x.endswith(".tmp.requirements.txt") for x in os.listdir())

    with pytest.raises(Exception):
        _parse_pip_requirements(["a", "-r does_not_exist.txt"])
    # Ensure the temporary requirements file has been removed even when parsing fails
    assert all(not x.endswith(".tmp.requirements.txt") for x in os.listdir())


@pytest.mark.parametrize("invalid_argument", [0, True, [0]])
def test_parse_pip_requirements_with_invalid_argument_types(invalid_argument):
    with pytest.raises(TypeError, match="`pip_requirements` must be either a string path"):
        _parse_pip_requirements(invalid_argument)


def test_validate_env_arguments():
    _validate_env_arguments(
        conda_env=None, pip_requirements=None, extra_pip_requirements=None,
    )

    match = "Only one of `conda_env`, `pip_requirements`, and `extra_pip_requirements`"
    with pytest.raises(ValueError, match=match):
        _validate_env_arguments(
            conda_env={}, pip_requirements=[], extra_pip_requirements=None,
        )

    with pytest.raises(ValueError, match=match):
        _validate_env_arguments(
            conda_env={}, pip_requirements=None, extra_pip_requirements=[],
        )

    with pytest.raises(ValueError, match=match):
        _validate_env_arguments(
            conda_env=None, pip_requirements=[], extra_pip_requirements=[],
        )

    with pytest.raises(ValueError, match=match):
        _validate_env_arguments(
            conda_env={}, pip_requirements=[], extra_pip_requirements=[],
        )
