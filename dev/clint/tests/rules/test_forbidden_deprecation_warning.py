from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules import ForbiddenDeprecationWarning


def test_forbidden_deprecation_warning(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
import warnings

# Bad - should be flagged
warnings.warn("message", category=DeprecationWarning)
warnings.warn(
    "multiline message",
    category=DeprecationWarning,
    stacklevel=2
)

# Good - should not be flagged
warnings.warn("message", category=FutureWarning)
warnings.warn("message", category=UserWarning)
warnings.warn("message")  # no category specified
warnings.warn("message", stacklevel=2)  # no category specified
other_function("message", category=DeprecationWarning)  # not warnings.warn
"""
    )

    config = Config(select={ForbiddenDeprecationWarning.name})
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 2
    assert all(isinstance(r.rule, ForbiddenDeprecationWarning) for r in results)
    assert results[0].loc == Location(4, 0)  # First warnings.warn call
    assert results[1].loc == Location(5, 0)  # Second warnings.warn call


def test_forbidden_deprecation_warning_import_variants(index_path: Path, tmp_path: Path) -> None:
    """Test detection with different import styles."""
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
import warnings
from warnings import warn
import warnings as w

# All of these should be flagged
warnings.warn("message", category=DeprecationWarning)
warn("message", category=DeprecationWarning)
w.warn("message", category=DeprecationWarning)
"""
    )

    config = Config(select={ForbiddenDeprecationWarning.name})
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 3
    assert all(isinstance(r.rule, ForbiddenDeprecationWarning) for r in results)


def test_forbidden_deprecation_warning_parameter_order(index_path: Path, tmp_path: Path) -> None:
    """Test detection regardless of parameter order."""
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
import warnings

# Different parameter orders - should be flagged
warnings.warn("message", category=DeprecationWarning)
warnings.warn(category=DeprecationWarning, message="test")
"""
    )

    config = Config(select={ForbiddenDeprecationWarning.name})
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 2
    assert all(isinstance(r.rule, ForbiddenDeprecationWarning) for r in results)


def test_forbidden_deprecation_warning_positional_args(index_path: Path, tmp_path: Path) -> None:
    """Test detection with positional arguments."""
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
import warnings

# Positional arguments - should be flagged
warnings.warn("message", DeprecationWarning)
warnings.warn("message", DeprecationWarning, 2)

# Good - should not be flagged
warnings.warn("message", FutureWarning)
warnings.warn("message")  # no category specified
"""
    )

    config = Config(select={ForbiddenDeprecationWarning.name})
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 2
    assert all(isinstance(r.rule, ForbiddenDeprecationWarning) for r in results)
