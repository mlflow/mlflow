from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules import SubprocessCheckCall


def test_subprocess_run_check_true_only(index_path: Path) -> None:
    """Test that subprocess.run with only check=True is flagged."""
    code = """
import subprocess

# Bad - only check=True
subprocess.run(["echo", "hello"], check=True)
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, SubprocessCheckCall)
    assert results[0].range == Range(Position(4, 0))


def test_subprocess_run_check_true_with_trailing_comma(index_path: Path) -> None:
    """Test that subprocess.run with check=True and trailing comma is flagged."""
    code = """
import subprocess

# Bad - check=True with trailing comma
subprocess.run(["echo", "hello"], check=True,)
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, SubprocessCheckCall)
    assert results[0].range == Range(Position(4, 0))


def test_subprocess_run_check_true_with_other_kwargs(index_path: Path) -> None:
    """Test that subprocess.run with check=True and other kwargs is NOT flagged."""
    code = """
import subprocess

# Good - check=True with text=True
subprocess.run(["echo", "hello"], check=True, text=True)

# Good - check=True with capture_output
subprocess.run(["echo", "hello"], check=True, capture_output=True)

# Good - check=True with cwd
subprocess.run(["echo", "hello"], check=True, cwd="/tmp")

# Good - check=True with env
subprocess.run(["echo", "hello"], check=True, env={})

# Good - check=True with stdout
subprocess.run(["echo", "hello"], check=True, stdout=subprocess.PIPE)
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_subprocess_check_call_not_flagged(index_path: Path) -> None:
    """Test that subprocess.check_call is NOT flagged."""
    code = """
import subprocess

# Good - already using check_call
subprocess.check_call(["echo", "hello"])
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_subprocess_run_without_check(index_path: Path) -> None:
    """Test that subprocess.run without check is NOT flagged."""
    code = """
import subprocess

# Good - no check parameter
subprocess.run(["echo", "hello"])

# Good - other parameters
subprocess.run(["echo", "hello"], text=True)
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_subprocess_run_check_false(index_path: Path) -> None:
    """Test that subprocess.run with check=False is NOT flagged."""
    code = """
import subprocess

# Good - check=False
subprocess.run(["echo", "hello"], check=False)
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_subprocess_run_with_alias(index_path: Path) -> None:
    """Test that subprocess.run with alias is correctly flagged."""
    code = """
import subprocess as sp

# Bad - with alias
sp.run(["echo", "hello"], check=True)
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, SubprocessCheckCall)
    assert results[0].range == Range(Position(4, 0))


def test_subprocess_run_from_import(index_path: Path) -> None:
    """Test that 'from subprocess import run' with check=True is flagged."""
    code = """
from subprocess import run

# Bad - from import with check=True
run(["echo", "hello"], check=True)
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, SubprocessCheckCall)
    assert results[0].range == Range(Position(4, 0))


def test_multiple_violations(index_path: Path) -> None:
    """Test that multiple violations are all caught."""
    code = """
import subprocess

# Bad - multiple calls with only check=True
subprocess.run(["echo", "hello"], check=True)
subprocess.run(["echo", "world"], check=True)
subprocess.run(["echo", "foo"], check=True)
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 3
    assert all(isinstance(r.rule, SubprocessCheckCall) for r in results)
    assert results[0].range == Range(Position(4, 0))
    assert results[1].range == Range(Position(5, 0))
    assert results[2].range == Range(Position(6, 0))


def test_subprocess_run_check_variable(index_path: Path) -> None:
    """Test that subprocess.run with check=variable is NOT flagged."""
    code = """
import subprocess

check_flag = True

# Good - check is a variable, not a literal
subprocess.run(["echo", "hello"], check=check_flag)
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_not_subprocess_run(index_path: Path) -> None:
    """Test that non-subprocess.run calls are NOT flagged."""
    code = """
class MyClass:
    def run(self, check=True):
        pass

obj = MyClass()

# Good - not subprocess.run
obj.run(check=True)
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0


def test_subprocess_run_mixed_scenarios(index_path: Path) -> None:
    """Test a mix of good and bad scenarios."""
    code = """
import subprocess

# Bad
subprocess.run(["ls"], check=True)

# Good - has other kwargs
subprocess.run(["ls"], check=True, text=True)

# Good - check_call
subprocess.check_call(["ls"])

# Bad
subprocess.run(["pwd"], check=True)

# Good - no check
subprocess.run(["pwd"])
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 2
    assert all(isinstance(r.rule, SubprocessCheckCall) for r in results)
    assert results[0].range == Range(Position(4, 0))
    assert results[1].range == Range(Position(13, 0))


def test_subprocess_run_with_kwargs_dict(index_path: Path) -> None:
    """Test that subprocess.run with **kwargs is NOT flagged."""
    code = """
import subprocess

kwargs = {"check": True}

# Good - using **kwargs
subprocess.run(["echo", "hello"], **kwargs)
"""
    config = Config(select={SubprocessCheckCall.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0
