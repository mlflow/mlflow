from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules import UnnamedThread


def test_unnamed_thread(index: SymbolIndex, config: Config, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
import threading

# Bad
threading.Thread(target=lambda: None)

# Good
# threading.Thread(target=lambda: None, name="worker")
"""
    )
    results = lint_file(tmp_file, config, index)
    assert len(results) == 1
    assert isinstance(results[0].rule, UnnamedThread)
    assert results[0].loc == Location(4, 0)
