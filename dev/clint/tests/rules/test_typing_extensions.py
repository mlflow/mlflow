from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules.typing_extensions import TypingExtensions


def test_typing_extensions(index: SymbolIndex, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
# Bad
from typing_extensions import ParamSpec

# Good
from typing_extensions import Self
"""
    )

    violations = lint_file(
        tmp_file,
        config=Config(typing_extensions_allowlist=["typing_extensions.Self"]),
        index=index,
    )
    assert len(violations) == 1
    assert all(isinstance(v.rule, TypingExtensions) for v in violations)
    assert violations[0].loc == Location(2, 0)
