from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.incorrect_type_annotation import IncorrectTypeAnnotation


def test_incorrect_type_annotation(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
def bad_function_callable(param: callable) -> callable:
    ...

def bad_function_any(param: any) -> any:
    ...

def good_function(param: Callable[[str], str]) -> Any:
    ...
"""
    )

    config = Config(select={IncorrectTypeAnnotation.name})
    violations = lint_file(tmp_file, config, index_path)
    assert len(violations) == 4
    assert all(isinstance(v.rule, IncorrectTypeAnnotation) for v in violations)
    assert violations[0].loc == Location(1, 33)  # callable
    assert violations[1].loc == Location(1, 46)  # callable
    assert violations[2].loc == Location(4, 28)  # any
    assert violations[3].loc == Location(4, 36)  # any
