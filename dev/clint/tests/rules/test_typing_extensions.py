from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.typing_extensions import TypingExtensions


def test_typing_extensions(index_path: Path) -> None:
    code = """
# Bad
from typing_extensions import ParamSpec

# Good
from typing_extensions import Self
"""
    config = Config(
        select={TypingExtensions.name}, typing_extensions_allowlist=["typing_extensions.Self"]
    )
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, TypingExtensions) for v in violations)
    assert violations[0].loc == Location(2, 0)
