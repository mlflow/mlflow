from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules import UnnamedThreadPool


def test_thread_pool_executor(index_path: Path) -> None:
    code = """
from concurrent.futures import ThreadPoolExecutor

# Bad
ThreadPoolExecutor()

# Good
ThreadPoolExecutor(thread_name_prefix="worker")
"""
    config = Config(select={UnnamedThreadPool.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, UnnamedThreadPool)
    assert results[0].range == Range(Position(4, 0))
