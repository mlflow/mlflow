from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules import ThreadPoolExecutorWithoutThreadNamePrefix


def test_thread_pool_executor(index: SymbolIndex, config: Config, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
from concurrent.futures import ThreadPoolExecutor

# Bad
ThreadPoolExecutor()

# Good
ThreadPoolExecutor(thread_name_prefix="worker")
"""
    )
    results = lint_file(tmp_file, config, index)
    assert len(results) == 1
    assert isinstance(results[0].rule, ThreadPoolExecutorWithoutThreadNamePrefix)
    assert results[0].loc == Location(4, 0)
