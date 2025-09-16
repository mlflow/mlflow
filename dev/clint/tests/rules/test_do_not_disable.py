from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.do_not_disable import DoNotDisable


def test_do_not_disable(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
# Bad B006
# noqa: B006

# Bad F821
# noqa: F821

# Good
# noqa: B004
"""
    )

    config = Config(select={DoNotDisable.name})
    violations = lint_file(tmp_file, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, DoNotDisable) for v in violations)
    assert violations[0].loc == Location(2, 0)
    assert violations[1].loc == Location(5, 0)
