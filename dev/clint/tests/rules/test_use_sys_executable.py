from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules import UseSysExecutable


def test_use_sys_executable(index: SymbolIndex, config: Config, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
import subprocess
import sys

# Bad
subprocess.run(["mlflow", "ui"])
subprocess.check_call(["mlflow", "ui"])

# Good
subprocess.run([sys.executable, "-m", "mlflow", "ui"])
subprocess.check_call([sys.executable, "-m", "mlflow", "ui"])
"""
    )
    results = lint_file(tmp_file, config, index)
    assert len(results) == 2
    assert all(isinstance(r.rule, UseSysExecutable) for r in results)
    assert results[0].loc == Location(5, 0)
    assert results[1].loc == Location(6, 0)
