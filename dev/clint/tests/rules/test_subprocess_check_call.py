from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules import SubprocessCheckCall


def test_subprocess_check_call(index_path: Path) -> None:
    code = """
import subprocess

# Bad
subprocess.run(["echo", "hello"], check=True)

# Good - has other kwargs
subprocess.run(["echo", "hello"], check=True, text=True)

# Good - check_call
subprocess.check_call(["echo", "hello"])

# Good - no check
subprocess.run(["echo", "hello"])
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, SubprocessCheckCall)
    assert results[0].range == Range(Position(4, 0))
