from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules import UnnamedThread


def test_unnamed_thread(index_path: Path) -> None:
    code = """
import threading

# Bad
threading.Thread(target=lambda: None)

# Good
# threading.Thread(target=lambda: None, name="worker")
"""
    config = Config(select={UnnamedThread.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UnnamedThread)
    assert results[0].loc == Location(4, 0)
