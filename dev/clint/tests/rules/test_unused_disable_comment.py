from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules import UnusedDisableComment


def test_stale_disable_comment(index_path: Path) -> None:
    code = """
import os  # clint: disable=lazy-builtin-import
"""
    config = Config(select={UnusedDisableComment.name, "lazy-builtin-import"})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UnusedDisableComment)
    assert results[0].rule.rule_name == "lazy-builtin-import"
    assert results[0].range == Range(Position(1, 13))


def test_active_disable_comment(index_path: Path) -> None:
    code = """
def f():
    import os  # clint: disable=lazy-builtin-import
"""
    config = Config(select={UnusedDisableComment.name, "lazy-builtin-import"})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_mixed_disable_comments(index_path: Path) -> None:
    code = """
import os  # clint: disable=lazy-builtin-import

def f():
    import sys  # clint: disable=lazy-builtin-import
"""
    config = Config(select={UnusedDisableComment.name, "lazy-builtin-import"})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UnusedDisableComment)
    assert results[0].rule.rule_name == "lazy-builtin-import"
    assert results[0].range == Range(Position(1, 13))


def test_unused_disable_comment_can_be_disabled(index_path: Path) -> None:
    code = """
import os  # clint: disable=lazy-builtin-import,unused-disable-comment
"""
    config = Config(select={UnusedDisableComment.name, "lazy-builtin-import"})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0
