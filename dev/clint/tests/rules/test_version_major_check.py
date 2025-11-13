from pathlib import Path

from clint.config import Config
from clint.linter import lint_file
from clint.rules.version_major_check import MajorVersionCheck


def test_version_major_check(index_path: Path) -> None:
    code = """
from packaging.version import Version

Version("0.9.0") >= Version("1.0.0")
Version("1.2.3").major >= 1
Version("1.0.0") >= Version("0.83.0")
Version("1.5.0") >= Version("2.0.0")
Version("1.5.0") == Version("3.0.0")
Version("1.5.0") != Version("4.0.0")
"""
    config = Config(select={MajorVersionCheck.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 4
    assert all(isinstance(v.rule, MajorVersionCheck) for v in violations)
    assert violations[0].range.start.line == 3
    assert violations[1].range.start.line == 6
    assert violations[2].range.start.line == 7
    assert violations[3].range.start.line == 8


def test_version_major_check_no_violations(index_path: Path) -> None:
    code = """
from packaging.version import Version

Version("1.2.3").major >= 1
Version("1.0.0") >= Version("0.83.0")
Version("1.5.0") >= Version("1.0.1")
Version("1.5.0") >= Version("1.0.0.dev0")
5 >= 3
"""
    config = Config(select={MajorVersionCheck.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 0
