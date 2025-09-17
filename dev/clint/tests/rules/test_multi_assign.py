from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules import MultiAssign


def test_multi_assign(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    code = """
# Bad
x, y = 1, 2

# Good
a, b = func()
"""
    tmp_file.write_text(code)
    config = Config(select={MultiAssign.name})
    results = lint_file(tmp_file, code, config, index_path)
    assert len(results) == 1
    assert all(isinstance(r.rule, MultiAssign) for r in results)
    assert results[0].loc == Location(2, 0)
