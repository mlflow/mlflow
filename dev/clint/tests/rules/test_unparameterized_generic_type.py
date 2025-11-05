from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules.unparameterized_generic_type import UnparameterizedGenericType


def test_unparameterized_generic_type(index_path: Path) -> None:
    code = """
from typing import Callable, Sequence

# Bad - unparameterized built-in types
def bad_list() -> list:
    pass

def bad_dict() -> dict:
    pass

# Good - parameterized built-in types
def good_list() -> list[str]:
    pass

def good_dict() -> dict[str, int]:
    pass
"""
    config = Config(select={UnparameterizedGenericType.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, UnparameterizedGenericType) for v in violations)
    assert violations[0].range == Range(Position(4, 18))  # bad_list return type
    assert violations[1].range == Range(Position(7, 18))  # bad_dict return type
