from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import lint_file
from clint.rules.unsafe_version_parse import UnsafeVersionParse


def test_unsafe_version_parse(index: SymbolIndex) -> None:
    code = """
import importlib.metadata
import importlib_metadata
from packaging.version import Version

Version(importlib.metadata.version("some-dist"))
Version(importlib_metadata.version("some-dist"))
Version(runtime_version)
Version(dbr_version)
Version(udf_sandbox_info.runtime_version)
Version(info.databricks_runtime)
"""
    config = Config(select={UnsafeVersionParse.name})
    violations = lint_file(Path("test.py"), code, config, index)
    assert len(violations) == 6
    assert all(isinstance(v.rule, UnsafeVersionParse) for v in violations)
    assert violations[0].range.start.line == 5
    assert violations[1].range.start.line == 6
    assert violations[2].range.start.line == 7
    assert violations[3].range.start.line == 8
    assert violations[4].range.start.line == 9
    assert violations[5].range.start.line == 10


def test_unsafe_version_parse_no_violations(index: SymbolIndex) -> None:
    code = """
import importlib.metadata
from packaging.version import Version

from mlflow.utils import get_installed_version
from mlflow.utils.databricks_utils import parse_dbr_runtime_major_minor

# The safe helpers themselves are fine.
raw = importlib.metadata.version("some-dist")
Version(raw)
Version(__version__)
Version("1.2.3")
Version(module.__version__)
get_installed_version("some-dist")
parse_dbr_runtime_major_minor(runtime_version)
# A non-Version call wrapping metadata.version is not our concern.
str(importlib.metadata.version("some-dist"))
# An attribute unrelated to DBR runtime versions.
Version(config.some_version)
"""
    config = Config(select={UnsafeVersionParse.name})
    violations = lint_file(Path("test.py"), code, config, index)
    assert len(violations) == 0


def test_unsafe_version_parse_aliased_import(index: SymbolIndex) -> None:
    code = """
from importlib import metadata
from packaging.version import Version

Version(metadata.version("some-dist"))
"""
    config = Config(select={UnsafeVersionParse.name})
    violations = lint_file(Path("test.py"), code, config, index)
    assert len(violations) == 1
    assert violations[0].range.start.line == 4
