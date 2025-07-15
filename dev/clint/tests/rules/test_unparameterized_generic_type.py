from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules.unparameterized_generic_type import UnparameterizedGenericType


def test_unparameterized_generic_type(index: SymbolIndex, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
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
    )

    config = Config(select={UnparameterizedGenericType.name})
    violations = lint_file(tmp_file, config, index)
    assert len(violations) == 2
    assert all(isinstance(v.rule, UnparameterizedGenericType) for v in violations)
    assert violations[0].loc == Location(4, 18)  # bad_list return type
    assert violations[1].loc == Location(7, 18)  # bad_dict return type
