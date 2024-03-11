import difflib
import os
import re
from pathlib import Path

import pytest
from packaging.version import Version

from dev.update_mlflow_versions import (
    get_current_py_version,
    replace_java,
    replace_java_pom_xml,
    replace_js,
    replace_pyproject_toml,
    replace_python,
    replace_r,
)

# { filename: expected lines changed }
_JAVA_FILES = {
    "mlflow/java/scoring/src/main/java/org/mlflow/sagemaker/ScoringServer.java": [175],
    "mlflow/java/scoring/src/test/java/org/mlflow/ScoringServerTest.java": [81],
}

_JAVA_XML_FILES = {
    "mlflow/java/pom.xml": [6, 62],
    "mlflow/java/scoring/pom.xml": [8],
    "mlflow/java/client/pom.xml": [8],
    "mlflow/java/spark/pom.xml": [4, 19],
}

_JS_FILES = {
    "mlflow/server/js/src/common/constants.tsx": [12],
}

_PYTHON_FILES = {
    "mlflow/version.py": [4],
}

_PYPROJECT_TOML_FILES = {
    "pyproject.skinny.toml": [7],
    "pyproject.toml": [7],
}

_R_FILES = {
    "mlflow/R/mlflow/DESCRIPTION": [4],
}

_DIFF_REGEX = re.compile(r"--- (\d+) ----")

old_version = Version(get_current_py_version())
_NEW_PY_VERSION = f"{old_version.major}.{old_version.minor}.{old_version.micro + 1}"


def copy_and_run_change_func(tmp_path, paths_to_copy, replace_func, new_version):
    for path in paths_to_copy:
        copy_path = tmp_path / path
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_text(path.read_text())

    cwd = os.getcwd()
    os.chdir(tmp_path)
    replace_func(new_version, paths_to_copy)
    os.chdir(cwd)


@pytest.mark.parametrize(
    ("replace_func", "expect_dict", "new_py_version", "expected_new_version"),
    [
        (replace_java, _JAVA_FILES, _NEW_PY_VERSION, _NEW_PY_VERSION),
        (replace_java, _JAVA_FILES, _NEW_PY_VERSION + ".dev0", _NEW_PY_VERSION + "-SNAPSHOT"),
        (replace_java, _JAVA_FILES, _NEW_PY_VERSION + "rc1", _NEW_PY_VERSION + "-SNAPSHOT"),
        (replace_java_pom_xml, _JAVA_XML_FILES, _NEW_PY_VERSION, _NEW_PY_VERSION),
        (
            replace_java_pom_xml,
            _JAVA_XML_FILES,
            _NEW_PY_VERSION + ".dev0",
            _NEW_PY_VERSION + "-SNAPSHOT",
        ),
        (
            replace_java_pom_xml,
            _JAVA_XML_FILES,
            _NEW_PY_VERSION + "rc1",
            _NEW_PY_VERSION + "-SNAPSHOT",
        ),
        (replace_js, _JS_FILES, _NEW_PY_VERSION, _NEW_PY_VERSION),
        (replace_python, _PYTHON_FILES, _NEW_PY_VERSION, _NEW_PY_VERSION),
        (replace_pyproject_toml, _PYPROJECT_TOML_FILES, _NEW_PY_VERSION, _NEW_PY_VERSION),
        (replace_r, _R_FILES, _NEW_PY_VERSION, _NEW_PY_VERSION),
    ],
)
def test_update_mlflow_versions(
    tmp_path, replace_func, expect_dict, new_py_version, expected_new_version
):
    paths_to_change = [Path(filename) for filename in expect_dict]
    copy_and_run_change_func(
        tmp_path,
        # always copy version.py since we need it in get_current_py_version()
        paths_to_change + [Path("mlflow/version.py")],
        replace_func,
        new_py_version,
    )

    # diff files
    for filename, expected_lines_changed in expect_dict.items():
        old_file = Path(filename).read_text().split("\n")
        new_file = (tmp_path / filename).read_text().split("\n")
        diffs = list(difflib.context_diff(old_file, new_file, n=0))
        changed_lines = [
            int(_DIFF_REGEX.search(d).group(1)) for d in diffs if _DIFF_REGEX.search(d)
        ]

        assert changed_lines == expected_lines_changed

        for line in expected_lines_changed:
            assert expected_new_version in new_file[line - 1]
