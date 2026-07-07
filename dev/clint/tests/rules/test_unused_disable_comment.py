from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Position, Range, lint_file
from clint.rules import UnusedDisableComment


def test_stale_disable_comment(index: SymbolIndex) -> None:
    code = """
import os  # clint: disable=lazy-import
"""
    config = Config(select={UnusedDisableComment.name, "lazy-import"})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 1
    assert isinstance(results[0].rule, UnusedDisableComment)
    assert results[0].rule.rule_name == "lazy-import"
    assert results[0].range == Range(Position(1, 13))


def test_active_disable_comment(index: SymbolIndex) -> None:
    code = """
def f():
    import os  # clint: disable=lazy-import
"""
    config = Config(select={UnusedDisableComment.name, "lazy-import"})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_mixed_disable_comments(index: SymbolIndex) -> None:
    code = """
import os  # clint: disable=lazy-import

def f():
    import sys  # clint: disable=lazy-import
"""
    config = Config(select={UnusedDisableComment.name, "lazy-import"})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 1
    assert isinstance(results[0].rule, UnusedDisableComment)
    assert results[0].rule.rule_name == "lazy-import"
    assert results[0].range == Range(Position(1, 13))


def test_unused_disable_comment_can_be_disabled(index: SymbolIndex) -> None:
    code = """
import os  # clint: disable=lazy-import,unused-disable-comment
"""
    config = Config(select={UnusedDisableComment.name, "lazy-import"})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_disable_next_suppresses_next_line(index: SymbolIndex) -> None:
    code = """
def f():
    # clint: disable-next=lazy-import
    import os
"""
    config = Config(select={UnusedDisableComment.name, "lazy-import"})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_disable_next_unused_reports_at_comment_line(index: SymbolIndex) -> None:
    code = """
# clint: disable-next=lazy-import
import os
"""
    config = Config(select={UnusedDisableComment.name, "lazy-import"})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 1
    assert isinstance(results[0].rule, UnusedDisableComment)
    assert results[0].rule.rule_name == "lazy-import"
    assert results[0].range == Range(Position(1, 2))


def test_disable_next_multi_rule_partial_used(index: SymbolIndex) -> None:
    code = """
def f():
    # clint: disable-next=lazy-import,unused-disable-comment
    import os
"""
    config = Config(select={UnusedDisableComment.name, "lazy-import"})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 1
    assert isinstance(results[0].rule, UnusedDisableComment)
    assert results[0].rule.rule_name == "unused-disable-comment"
