from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules.do_not_disable import DoNotDisable


def test_do_not_disable(index_path: Path) -> None:
    code = """
# Bad B006
# noqa: B006

# Bad F821
# noqa: F821

# Good
# noqa: B004
"""
    config = Config(select={DoNotDisable.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, DoNotDisable) for v in violations)
    assert violations[0].range == Range(Position(2, 0))
    assert violations[1].range == Range(Position(5, 0))


def test_do_not_disable_comma_separated(index_path: Path) -> None:
    code = """
# Bad: B006 and F821 both should be caught
# noqa: B006, F821

# Bad: B006 and F821 both should be caught (no space after comma)
# noqa: B006,F821

# Good: B004 is allowed
# noqa: B004, B005
"""
    config = Config(select={DoNotDisable.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, DoNotDisable) for v in violations)
    # Both violations should have both rules B006 and F821
    assert isinstance(violations[0].rule, DoNotDisable)
    assert isinstance(violations[1].rule, DoNotDisable)
    assert violations[0].rule.rules == {"B006", "F821"}
    assert violations[1].rule.rules == {"B006", "F821"}
    assert violations[0].range == Range(Position(2, 0))
    assert violations[1].range == Range(Position(5, 0))
