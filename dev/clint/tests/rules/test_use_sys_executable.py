from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules import UseSysExecutable


def test_use_sys_executable(index_path: Path) -> None:
    code = """
import subprocess
import sys

# Bad
subprocess.run(["mlflow", "ui"])
subprocess.check_call(["mlflow", "ui"])

# Good
subprocess.run([sys.executable, "-m", "mlflow", "ui"])
subprocess.check_call([sys.executable, "-m", "mlflow", "ui"])
"""
    config = Config(select={UseSysExecutable.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 2
    assert all(isinstance(r.rule, UseSysExecutable) for r in results)
    assert results[0].loc == Location(5, 0)
    assert results[1].loc == Location(6, 0)
