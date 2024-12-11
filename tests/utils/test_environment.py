import importlib.metadata
import os
from unittest import mock

import pytest
import yaml

from mlflow.exceptions import MlflowException
from mlflow.utils.environment import (
    _contains_mlflow_requirement,
    _deduplicate_requirements,
    _get_pip_deps,
    _get_pip_requirement_specifier,
    _is_mlflow_requirement,
    _is_pip_deps,
    _mlflow_conda_env,
    _overwrite_pip_deps,
    _parse_pip_requirements,
    _process_conda_env,
    _process_pip_requirements,
    _validate_env_arguments,
    infer_pip_requirements,
)

from tests.helper_functions import _mlflow_major_version_string


@pytest.fixture
def conda_env_path(tmp_path):
    return os.path.join(tmp_path, "conda_env.yaml")


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
    pip_version = importlib.metadata.version("pip")
    assert f"pip<={pip_version}" in env["dependencies"]


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


def test_parse_pip_requirements(tmp_path):
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
    fake_whl = tmp_path.joinpath("fake.whl")
    fake_whl.write_text("")
    assert _parse_pip_requirements([str(fake_whl)]) == ([str(fake_whl)], [])


def test_parse_pip_requirements_with_relative_requirements_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f1 = tmp_path.joinpath("requirements1.txt")
    f1.write_text("b")
    assert _parse_pip_requirements(f1.name) == (["b"], [])
    assert _parse_pip_requirements(["a", f"-r {f1.name}"]) == (["a", "b"], [])

    f2 = tmp_path.joinpath("requirements2.txt")
    f3 = tmp_path.joinpath("requirements3.txt")
    f2.write_text(f"b\n-r {f3.name}")
    f3.write_text("c")
    assert _parse_pip_requirements(f2.name) == (["b", "c"], [])
    assert _parse_pip_requirements(["a", f"-r {f2.name}"]) == (["a", "b", "c"], [])


def test_parse_pip_requirements_with_absolute_requirements_files(tmp_path):
    f1 = tmp_path.joinpath("requirements1.txt")
    f1.write_text("b")
    assert _parse_pip_requirements(str(f1)) == (["b"], [])
    assert _parse_pip_requirements(["a", f"-r {f1}"]) == (["a", "b"], [])

    f2 = tmp_path.joinpath("requirements2.txt")
    f3 = tmp_path.joinpath("requirements3.txt")
    f2.write_text(f"b\n-r {f3}")
    f3.write_text("c")
    assert _parse_pip_requirements(str(f2)) == (["b", "c"], [])
    assert _parse_pip_requirements(["a", f"-r {f2}"]) == (["a", "b", "c"], [])


def test_parse_pip_requirements_with_constraints_files(tmp_path):
    con_file = tmp_path.joinpath("constraints.txt")
    con_file.write_text("b")
    assert _parse_pip_requirements(["a", f"-c {con_file}"]) == (["a"], ["b"])

    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text(f"-c {con_file}\n")
    assert _parse_pip_requirements(["a", f"-r {req_file}"]) == (["a"], ["b"])


def test_parse_pip_requirements_ignores_comments_and_blank_lines(tmp_path):
    reqs = [
        "# comment",
        "a # inline comment",
        # blank lines
        "",
        " ",
    ]
    f = tmp_path.joinpath("requirements.txt")
    f.write_text("\n".join(reqs))
    assert _parse_pip_requirements(reqs) == (["a"], [])
    assert _parse_pip_requirements(str(f)) == (["a"], [])


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
    assert _is_mlflow_requirement("mlflow-skinny==1.2.3")
    assert not _is_mlflow_requirement("foo")
    # Ensure packages that look like mlflow are NOT considered as mlflow.
    assert not _is_mlflow_requirement("mlflow-foo")
    assert not _is_mlflow_requirement("mlflow_foo")


def test_contains_mlflow_requirement():
    assert _contains_mlflow_requirement(["mlflow"])
    assert _contains_mlflow_requirement(["mlflow==1.2.3"])
    assert _contains_mlflow_requirement(["mlflow", "foo"])
    assert _contains_mlflow_requirement(["mlflow-skinny"])
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
    assert _get_pip_requirement_specifier("mlflow-skinny==1.2 --foo=bar") == "mlflow-skinny==1.2"


def test_process_pip_requirements(tmp_path):
    expected_mlflow_ver = _mlflow_major_version_string()
    conda_env, reqs, cons = _process_pip_requirements(["a"])
    assert _get_pip_deps(conda_env) == [expected_mlflow_ver, "a"]
    assert reqs == [expected_mlflow_ver, "a"]
    assert cons == []

    conda_env, reqs, cons = _process_pip_requirements(["a"], pip_requirements=["b"])
    assert _get_pip_deps(conda_env) == [expected_mlflow_ver, "b"]
    assert reqs == [expected_mlflow_ver, "b"]
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
    assert _get_pip_deps(conda_env) == [expected_mlflow_ver, "a", "b"]
    assert reqs == [expected_mlflow_ver, "a", "b"]
    assert cons == []

    con_file = tmp_path.joinpath("constraints.txt")
    con_file.write_text("c")
    conda_env, reqs, cons = _process_pip_requirements(
        ["a"], pip_requirements=["b", f"-c {con_file}"]
    )
    assert _get_pip_deps(conda_env) == [expected_mlflow_ver, "b", "-c constraints.txt"]
    assert reqs == [expected_mlflow_ver, "b", "-c constraints.txt"]
    assert cons == ["c"]

    conda_env, reqs, cons = _process_pip_requirements(["a"], extra_pip_requirements=["a[extras]"])
    assert _get_pip_deps(conda_env) == [expected_mlflow_ver, "a[extras]"]
    assert reqs == [expected_mlflow_ver, "a[extras]"]
    assert cons == []

    conda_env, reqs, cons = _process_pip_requirements(
        ["mlflow==1.2.3", "b[extra1]", "a==1.2.3"],
        extra_pip_requirements=["b[extra2]", "a[extras]"],
    )
    assert _get_pip_deps(conda_env) == ["mlflow==1.2.3", "b[extra1,extra2]", "a[extras]==1.2.3"]
    assert reqs == ["mlflow==1.2.3", "b[extra1,extra2]", "a[extras]==1.2.3"]
    assert cons == []


def test_process_conda_env(tmp_path):
    def make_conda_env(pip_deps):
        return {
            "name": "mlflow-env",
            "channels": ["conda-forge"],
            "dependencies": ["python=3.8.15", "pip", {"pip": pip_deps}],
        }

    expected_mlflow_ver = _mlflow_major_version_string()

    conda_env, reqs, cons = _process_conda_env(make_conda_env(["a"]))
    assert _get_pip_deps(conda_env) == [expected_mlflow_ver, "a"]
    assert reqs == [expected_mlflow_ver, "a"]
    assert cons == []

    conda_env_file = tmp_path.joinpath("conda_env.yaml")
    conda_env_file.write_text(yaml.dump(make_conda_env(["a"])))
    conda_env, reqs, cons = _process_conda_env(str(conda_env_file))
    assert _get_pip_deps(conda_env) == [expected_mlflow_ver, "a"]
    assert reqs == [expected_mlflow_ver, "a"]
    assert cons == []

    # Ensure a requirement for mlflow is preserved
    conda_env, reqs, cons = _process_conda_env(make_conda_env(["mlflow==1.2.3"]))
    assert _get_pip_deps(conda_env) == ["mlflow==1.2.3"]
    assert reqs == ["mlflow==1.2.3"]
    assert cons == []

    con_file = tmp_path.joinpath("constraints.txt")
    con_file.write_text("c")
    conda_env, reqs, cons = _process_conda_env(make_conda_env(["a", f"-c {con_file}"]))
    assert _get_pip_deps(conda_env) == [expected_mlflow_ver, "a", "-c constraints.txt"]
    assert reqs == [expected_mlflow_ver, "a", "-c constraints.txt"]
    assert cons == ["c"]

    # NB: mlflow-skinny is not automatically attached to any model. If specified, it is
    # up to the user to pin a version.
    conda_env, reqs, cons = _process_conda_env(make_conda_env(["mlflow-skinny", "a", "b"]))
    assert _get_pip_deps(conda_env) == ["mlflow-skinny", "a", "b"]
    assert reqs == ["mlflow-skinny", "a", "b"]
    assert cons == []

    with pytest.raises(TypeError, match=r"Expected .+, but got `int`"):
        _process_conda_env(0)


@pytest.mark.parametrize(
    ("env_var", "fallbacks", "should_raise"),
    [
        # 1&2. If env var is True, always throw an exception from inference error
        (True, ["sklearn"], True),
        (True, None, True),
        # 3. If env var is False but fallback is provided, should not throw an exception
        (False, ["sklearn"], False),
        # 4. If fallback is not provided, should throw an exception
        (False, None, True),
    ],
)
def test_infer_requirements_error_handling(env_var, fallbacks, should_raise, monkeypatch):
    monkeypatch.setenv("MLFLOW_REQUIREMENTS_INFERENCE_RAISE_ERRORS", str(env_var))

    call_args = ("path/to/model", "sklearn", fallbacks)
    with mock.patch(
        "mlflow.utils.requirements_utils._capture_imported_modules",
        side_effect=MlflowException("Failed to capture imported modules"),
    ):
        if should_raise:
            with pytest.raises(MlflowException, match="Failed to capture imported module"):
                infer_pip_requirements(*call_args)
        else:
            # Should just pass with warning
            with mock.patch("mlflow.utils.environment._logger.warning") as warning_mock:
                infer_pip_requirements(*call_args)
            warning_mock.assert_called_once()
            warning_text = warning_mock.call_args[0][0]
            assert "Encountered an unexpected error while inferring" in warning_text


@pytest.mark.parametrize(
    ("input_requirements", "expected"),
    [
        # Simple cases
        (["scikit-learn>1", "pandas"], ["scikit-learn>1", "pandas"]),
        # Duplicates without extras, preserving version restrictions
        (["packageA", "packageA==1.0"], ["packageA==1.0"]),
        # Duplicates with extras
        (["packageA", "packageA[extras]"], ["packageA[extras]"]),
        # Mixed cases
        (
            ["packageA", "packageB", "packageA[extras]", "packageC<=2.0"],
            ["packageA[extras]", "packageB", "packageC<=2.0"],
        ),
        # Mixed versions and extras
        (["markdown>=3.5.1", "markdown[extras]", "markdown<4"], ["markdown[extras]>=3.5.1,<4"]),
        # Overlapping extras
        (
            ["packageZ[extra1]", "packageZ[extra2]", "packageZ"],
            ["packageZ[extra1,extra2]"],
        ),
        # No version on extras with final version on non-extras
        (
            ["markdown[extra1]", "markdown[extra2]", "markdown>3", "markdown<4"],
            ["markdown[extra1,extra2]>3,<4"],
        ),
        # Version constraints with extras
        (["markdown>1.0", "markdown[extras]<4"], ["markdown[extras]>1.0,<4"]),
        # Verify duplicate specifiers are not preserved
        (
            ["markdown==3.5.1", "markdown[extras]==3.5.1", "markdown[extras]"],
            ["markdown[extras]==3.5.1"],
        ),
        # Verify duplicate extras are not preserved
        (["markdown[extras]", "markdown", "markdown[extras]"], ["markdown[extras]"]),
    ],
)
def test_deduplicate_requirements_resolve_correctly(input_requirements, expected):
    assert _deduplicate_requirements(input_requirements) == expected


@pytest.mark.parametrize(
    "input_requirements",
    [
        # Non-inclusive range with precise specifier
        ["scikit-learn==1.1", "scikit-learn<1"],
        # Incompatible ranges with extras
        ["markdown[extras]==3.5.1", "markdown<3.4"],
        # Invalid ranges
        ["markdown<3", "markdown>3"],
        # Conflicting versions
        ["markdown==3.0", "markdown==3.5"],
    ],
)
def test_invalid_requirements_raise(input_requirements):
    with pytest.raises(
        MlflowException, match="The specified requirements versions are incompatible"
    ):
        _deduplicate_requirements(input_requirements)
