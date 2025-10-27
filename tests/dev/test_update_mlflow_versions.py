import difflib
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
_JAVA_FILES = {}

_JAVA_XML_FILES = {
    "mlflow/java/pom.xml": {
        6: "  <version>{new_version}</version>",
        52: "    <mlflow-version>{new_version}</mlflow-version>",
    },
    "mlflow/java/client/pom.xml": {
        8: "    <version>{new_version}</version>",
    },
    "mlflow/java/spark_2.12/pom.xml": {
        4: "  <version>{new_version}</version>",
        18: "    <version>{new_version}</version>",
    },
    "mlflow/java/spark_2.13/pom.xml": {
        4: "  <version>{new_version}</version>",
        18: "    <version>{new_version}</version>",
    },
}

_JS_FILES = {
    "mlflow/server/js/src/common/constants.tsx": {
        12: "export const Version = '{new_version}';",
    }
}

_PYTHON_FILES = {
    "mlflow/version.py": {
        5: 'VERSION = "{new_version}"',
    }
}

_PYPROJECT_TOML_FILES = {
    "pyproject.toml": {
        12: 'version = "{new_version}"',
    },
    "pyproject.release.toml": {
        12: 'version = "{new_version}"',
        30: '  "mlflow-skinny=={new_version}",',
        31: '  "mlflow-tracing=={new_version}",',
    },
    "libs/skinny/pyproject.toml": {
        10: 'version = "{new_version}"',
    },
    "libs/tracing/pyproject.toml": {
        10: 'version = "{new_version}"',
    },
}

_R_FILES = {
    "mlflow/R/mlflow/DESCRIPTION": {
        4: "Version: {new_version}",
    }
}

_DIFF_REGEX = re.compile(r"--- (\d+,?\d*) ----")

old_version = Version(get_current_py_version())
_NEW_PY_VERSION = f"{old_version.major}.{old_version.minor}.{old_version.micro + 1}"


def copy_and_run_change_func(monkeypatch, tmp_path, paths_to_copy, replace_func, new_version):
    for path in paths_to_copy:
        copy_path = tmp_path / path
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_text(path.read_text())

    with monkeypatch.context() as m:
        m.chdir(tmp_path)

        # pyproject.toml replace doesn't search for the old version,
        # it just replaces the version line with the new version.
        if replace_func == replace_pyproject_toml:
            replace_func(new_version, paths_to_copy)
        else:
            replace_func(str(old_version), new_version, paths_to_copy)


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
    monkeypatch, tmp_path, replace_func, expect_dict, new_py_version, expected_new_version
):
    paths_to_change = [Path(filename) for filename in expect_dict]
    copy_and_run_change_func(
        monkeypatch,
        tmp_path,
        # always copy version.py since we need it in get_current_py_version()
        paths_to_change + [Path("mlflow/version.py")],
        replace_func,
        new_py_version,
    )

    # diff files
    for filename, expected_changes in expect_dict.items():
        old_file = Path(filename).read_text().splitlines()
        new_file = (tmp_path / filename).read_text().splitlines()
        diff = list(difflib.context_diff(old_file, new_file, n=0))
        changed_lines = _parse_diff_line(diff)

        formatted_expected_changes = {
            line_num: change.format(new_version=expected_new_version)
            for line_num, change in expected_changes.items()
        }

        assert changed_lines == formatted_expected_changes


def _parse_diff_line(diff: list[str]) -> dict[int, str]:
    diff_lines = {}
    for idx, line in enumerate(diff):
        match = _DIFF_REGEX.search(line)
        if not match:
            continue

        if "," in match.group(1):
            # multi-line change is represented as [(start,end), line1, line2, ...]
            start, end = map(int, match.group(1).split(","))
            for i in range(start, end + 1):
                # the [2:] is to cut out the "! " at the beginning of diff lines
                diff_lines[i] = diff[idx + (i - start) + 1][2:]
        else:
            # single-line change
            diff_lines[int(match.group(1))] = diff[idx + 1][2:]

    return diff_lines
