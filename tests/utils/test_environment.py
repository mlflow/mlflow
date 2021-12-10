import os
import pytest
import yaml

from mlflow.utils.environment import (
    _mlflow_conda_env,
    _is_pip_deps,
    _get_pip_deps,
    _overwrite_pip_deps,
    _parse_pip_requirements,
    _validate_env_arguments,
    _is_mlflow_requirement,
    _contains_mlflow_requirement,
    _process_pip_requirements,
    _process_conda_env,
    _get_pip_requirement_specifier,
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


def test_is_pip_deps():
    assert _is_pip_deps({"pip": ["a"]})
    assert not _is_pip_deps({"ipi": ["a"]})
    assert not _is_pip_deps("")
    assert not _is_pip_deps([])


def test_overwrite_pip_deps():
    # dependencies field doesn't exist
    name_and_channels = {"name": "env", "channels": ["conda-forge"]}
    expected = {**name_and_channels, "dependencies": [{"pip": ["scipy"]}]}
    assert _overwrite_pip_deps(name_and_channels, ["scipy"]) == expected

    # dependencies field doesn't contain pip dependencies
    conda_env = {**name_and_channels, "dependencies": ["pip"]}
    expected = {**name_and_channels, "dependencies": ["pip", {"pip": ["scipy"]}]}
    assert _overwrite_pip_deps(conda_env, ["scipy"]) == expected

    # dependencies field contains pip dependencies
    conda_env = {**name_and_channels, "dependencies": ["pip", {"pip": ["numpy"]}, "pandas"]}
    expected = {**name_and_channels, "dependencies": ["pip", {"pip": ["scipy"]}, "pandas"]}
    assert _overwrite_pip_deps(conda_env, ["scipy"]) == expected


def test_parse_pip_requirements(tmpdir):
    assert _parse_pip_requirements(None) == ([], [])
    assert _parse_pip_requirements([]) == ([], [])
    # Without version specifiers
    assert _parse_pip_requirements(["a", "b"]) == (["a", "b"], [])
    # With version specifiers
    assert _parse_pip_requirements(["a==0.0", "b>1.1"]) == (["a==0.0", "b>1.1"], [])
    # Environment marker (https://www.python.org/dev/peps/pep-0508/#environment-markers)
    assert _parse_pip_requirements(['a; python_version < "3.8"']) == (
        ['a; python_version < "3.8"'],
        [],
    )
    # GitHub URI
    mlflow_repo_uri = "git+https://github.com/mlflow/mlflow.git"
    assert _parse_pip_requirements([mlflow_repo_uri]) == ([mlflow_repo_uri], [])
    # Local file
    fake_whl = tmpdir.join("fake.whl")
    fake_whl.write("")
    assert _parse_pip_requirements([fake_whl.strpath]) == ([fake_whl.strpath], [])


def test_parse_pip_requirements_with_relative_requirements_files(request, tmpdir):
    try:
        os.chdir(tmpdir)
        f1 = tmpdir.join("requirements1.txt")
        f1.write("b")
        assert _parse_pip_requirements(f1.basename) == (["b"], [])
        assert _parse_pip_requirements(["a", f"-r {f1.basename}"]) == (["a", "b"], [])

        f2 = tmpdir.join("requirements2.txt")
        f3 = tmpdir.join("requirements3.txt")
        f2.write(f"b\n-r {f3.basename}")
        f3.write("c")
        assert _parse_pip_requirements(f2.basename) == (["b", "c"], [])
        assert _parse_pip_requirements(["a", f"-r {f2.basename}"]) == (["a", "b", "c"], [])
    finally:
        os.chdir(request.config.invocation_dir)


def test_parse_pip_requirements_with_absolute_requirements_files(tmpdir):
    f1 = tmpdir.join("requirements1.txt")
    f1.write("b")
    assert _parse_pip_requirements(f1.strpath) == (["b"], [])
    assert _parse_pip_requirements(["a", f"-r {f1.strpath}"]) == (["a", "b"], [])

    f2 = tmpdir.join("requirements2.txt")
    f3 = tmpdir.join("requirements3.txt")
    f2.write(f"b\n-r {f3.strpath}")
    f3.write("c")
    assert _parse_pip_requirements(f2.strpath) == (["b", "c"], [])
    assert _parse_pip_requirements(["a", f"-r {f2.strpath}"]) == (["a", "b", "c"], [])


def test_parse_pip_requirements_with_constraints_files(tmpdir):
    con_file = tmpdir.join("constraints.txt")
    con_file.write("b")
    assert _parse_pip_requirements(["a", f"-c {con_file.strpath}"]) == (["a"], ["b"])

    req_file = tmpdir.join("requirements.txt")
    req_file.write(f"-c {con_file.strpath}\n")
    assert _parse_pip_requirements(["a", f"-r {req_file.strpath}"]) == (["a"], ["b"])


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
    assert _parse_pip_requirements(reqs) == (["a"], [])
    assert _parse_pip_requirements(f.strpath) == (["a"], [])


def test_parse_pip_requirements_removes_temporary_requirements_file():
    assert _parse_pip_requirements(["a"]) == (["a"], [])
    assert all(not x.endswith(".tmp.requirements.txt") for x in os.listdir())

    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        _parse_pip_requirements(["a", "-r does_not_exist.txt"])
    # Ensure the temporary requirements file has been removed even when parsing fails
    assert all(not x.endswith(".tmp.requirements.txt") for x in os.listdir())


@pytest.mark.parametrize("invalid_argument", [0, True, [0]])
def test_parse_pip_requirements_with_invalid_argument_types(invalid_argument):
    with pytest.raises(TypeError, match="`pip_requirements` must be either a string path"):
        _parse_pip_requirements(invalid_argument)


def test_validate_env_arguments():
    _validate_env_arguments(pip_requirements=None, extra_pip_requirements=None, conda_env=None)

    match = "Only one of `conda_env`, `pip_requirements`, and `extra_pip_requirements`"
    with pytest.raises(ValueError, match=match):
        _validate_env_arguments(conda_env={}, pip_requirements=[], extra_pip_requirements=None)

    with pytest.raises(ValueError, match=match):
        _validate_env_arguments(conda_env={}, pip_requirements=None, extra_pip_requirements=[])

    with pytest.raises(ValueError, match=match):
        _validate_env_arguments(conda_env=None, pip_requirements=[], extra_pip_requirements=[])

    with pytest.raises(ValueError, match=match):
        _validate_env_arguments(conda_env={}, pip_requirements=[], extra_pip_requirements=[])


def test_is_mlflow_requirement():
    assert _is_mlflow_requirement("mlflow")
    assert _is_mlflow_requirement("MLFLOW")
    assert _is_mlflow_requirement("MLflow")
    assert _is_mlflow_requirement("mlflow==1.2.3")
    assert _is_mlflow_requirement("mlflow < 1.2.3")
    assert _is_mlflow_requirement("mlflow; python_version < '3.8'")
    assert _is_mlflow_requirement("mlflow @ https://github.com/mlflow/mlflow.git")
    assert _is_mlflow_requirement("mlflow @ file:///path/to/mlflow")
    assert not _is_mlflow_requirement("foo")
    # Ensure packages that look like mlflow are NOT considered as mlflow.
    assert not _is_mlflow_requirement("mlflow-foo")
    assert not _is_mlflow_requirement("mlflow_foo")


def test_contains_mlflow_requirement():
    assert _contains_mlflow_requirement(["mlflow"])
    assert _contains_mlflow_requirement(["mlflow==1.2.3"])
    assert _contains_mlflow_requirement(["mlflow", "foo"])
    assert not _contains_mlflow_requirement([])
    assert not _contains_mlflow_requirement(["foo"])


def test_get_pip_requirement_specifier():
    assert _get_pip_requirement_specifier("") == ""
    assert _get_pip_requirement_specifier(" ") == " "
    assert _get_pip_requirement_specifier("mlflow") == "mlflow"
    assert _get_pip_requirement_specifier("mlflow==1.2.3") == "mlflow==1.2.3"
    assert _get_pip_requirement_specifier("-r reqs.txt") == ""
    assert _get_pip_requirement_specifier("  -r reqs.txt") == " "
    assert _get_pip_requirement_specifier("mlflow==1.2.3 --hash=foo") == "mlflow==1.2.3"
    assert _get_pip_requirement_specifier("mlflow==1.2.3       --hash=foo") == "mlflow==1.2.3      "


def test_process_pip_requirements(tmpdir):
    conda_env, reqs, cons = _process_pip_requirements(["a"])
    assert _get_pip_deps(conda_env) == ["mlflow", "a"]
    assert reqs == ["mlflow", "a"]
    assert cons == []

    conda_env, reqs, cons = _process_pip_requirements(["a"], pip_requirements=["b"])
    assert _get_pip_deps(conda_env) == ["mlflow", "b"]
    assert reqs == ["mlflow", "b"]
    assert cons == []

    # Ensure a requirement for mlflow is preserved
    conda_env, reqs, cons = _process_pip_requirements(["a"], pip_requirements=["mlflow==1.2.3"])
    assert _get_pip_deps(conda_env) == ["mlflow==1.2.3"]
    assert reqs == ["mlflow==1.2.3"]
    assert cons == []

    # Ensure a requirement for mlflow is preserved when package hashes are specified
    hash1 = "sha256:963c22532e82a93450674ab97d62f9e528ed0906b580fadb7c003e696197557c"
    hash2 = "sha256:b15ff0c7e5e64f864a0b40c99b9a582227315eca2065d9f831db9aeb8f24637b"
    conda_env, reqs, cons = _process_pip_requirements(
        ["a"],
        pip_requirements=[f"mlflow==1.20.2 --hash={hash1} --hash={hash2}"],
    )
    assert _get_pip_deps(conda_env) == [f"mlflow==1.20.2 --hash={hash1} --hash={hash2}"]
    assert reqs == [f"mlflow==1.20.2 --hash={hash1} --hash={hash2}"]
    assert cons == []

    conda_env, reqs, cons = _process_pip_requirements(["a"], extra_pip_requirements=["b"])
    assert _get_pip_deps(conda_env) == ["mlflow", "a", "b"]
    assert reqs == ["mlflow", "a", "b"]
    assert cons == []

    con_file = tmpdir.join("constraints.txt")
    con_file.write("c")
    conda_env, reqs, cons = _process_pip_requirements(
        ["a"], pip_requirements=["b", f"-c {con_file.strpath}"]
    )
    assert _get_pip_deps(conda_env) == ["mlflow", "b", "-c constraints.txt"]
    assert reqs == ["mlflow", "b", "-c constraints.txt"]
    assert cons == ["c"]


def test_process_conda_env(tmpdir):
    def make_conda_env(pip_deps):
        return {
            "name": "mlflow-env",
            "channels": ["conda-forge"],
            "dependencies": ["python=3.7.9", "pip", {"pip": pip_deps}],
        }

    conda_env, reqs, cons = _process_conda_env(make_conda_env(["a"]))
    assert _get_pip_deps(conda_env) == ["mlflow", "a"]
    assert reqs == ["mlflow", "a"]
    assert cons == []

    conda_env_file = tmpdir.join("conda_env.yaml")
    conda_env_file.write(yaml.dump(make_conda_env(["a"])))
    conda_env, reqs, cons = _process_conda_env(conda_env_file.strpath)
    assert _get_pip_deps(conda_env) == ["mlflow", "a"]
    assert reqs == ["mlflow", "a"]
    assert cons == []

    # Ensure a requirement for mlflow is preserved
    conda_env, reqs, cons = _process_conda_env(make_conda_env(["mlflow==1.2.3"]))
    assert _get_pip_deps(conda_env) == ["mlflow==1.2.3"]
    assert reqs == ["mlflow==1.2.3"]
    assert cons == []

    con_file = tmpdir.join("constraints.txt")
    con_file.write("c")
    conda_env, reqs, cons = _process_conda_env(make_conda_env(["a", f"-c {con_file.strpath}"]))
    assert _get_pip_deps(conda_env) == ["mlflow", "a", "-c constraints.txt"]
    assert reqs == ["mlflow", "a", "-c constraints.txt"]
    assert cons == ["c"]

    with pytest.raises(TypeError, match=r"Expected .+, but got `int`"):
        _process_conda_env(0)
