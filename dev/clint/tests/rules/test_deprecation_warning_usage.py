from pathlib import Path

from clint.config import Config
from clint.linter import lint_file
from clint.rules.deprecation_warning_usage import DeprecationWarningUsage


def test_deprecation_warning_usage(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
import warnings

# Bad - positional argument
warnings.warn("This is deprecated", DeprecationWarning)

# Bad - keyword argument
warnings.warn("This is deprecated", category=DeprecationWarning)

# Bad - different message styles
warnings.warn(
    "This is also deprecated",
    DeprecationWarning,
)

# Good - using FutureWarning
warnings.warn("This is deprecated", FutureWarning)

# Good - using category=FutureWarning
warnings.warn("This is deprecated", category=FutureWarning)

# Good - no category specified (defaults to UserWarning)
warnings.warn("This is a warning")

# Good - other warning types
warnings.warn("This is deprecated", UserWarning)
warnings.warn("This is deprecated", category=UserWarning)

# Edge case - with additional arguments
warnings.warn("This is deprecated", DeprecationWarning, stacklevel=2)

# Edge case - category keyword with additional args
warnings.warn("This is deprecated", category=DeprecationWarning, stacklevel=2)
"""
    )

    config = Config(select={DeprecationWarningUsage.name})
    violations = lint_file(tmp_file, config, index_path)

    # Should find 5 violations:
    # 1. Line 5: warnings.warn("This is deprecated", DeprecationWarning)
    # 2. Line 8: warnings.warn("This is deprecated", category=DeprecationWarning)
    # 3. Line 11-13: Multi-line warnings.warn with DeprecationWarning
    # 4. Line 30: warnings.warn("This is deprecated", DeprecationWarning, stacklevel=2)
    # 5. Line 33: warnings.warn("This is deprecated", category=DeprecationWarning, stacklevel=2)
    assert len(violations) == 5
    assert all(isinstance(v.rule, DeprecationWarningUsage) for v in violations)

    # Check locations (0-indexed, so subtract 1 from line numbers)
    expected_lines = [4, 7, 10, 29, 32]
    actual_lines = [v.loc.lineno for v in violations]
    assert actual_lines == expected_lines


def test_deprecation_warning_usage_no_violations(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
import warnings

# All good usage - should not trigger violations
warnings.warn("This is deprecated", FutureWarning)
warnings.warn("This is deprecated", category=FutureWarning)
warnings.warn("This is a warning")
warnings.warn("This is deprecated", UserWarning)
warnings.warn("This is deprecated", category=UserWarning)

# Not warnings.warn calls
print("DeprecationWarning")
some_function(DeprecationWarning)
"""
    )

    config = Config(select={DeprecationWarningUsage.name})
    violations = lint_file(tmp_file, config, index_path)
    assert len(violations) == 0


def test_deprecation_warning_usage_with_aliases(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
from warnings import warn

# These should be detected since we enhanced the rule
warn("This is deprecated", DeprecationWarning)
warn("This is deprecated", category=DeprecationWarning)
"""
    )

    config = Config(select={DeprecationWarningUsage.name})
    violations = lint_file(tmp_file, config, index_path)

    # Should detect both violations
    assert len(violations) == 2
    assert all(isinstance(v.rule, DeprecationWarningUsage) for v in violations)

    # Check locations (0-indexed, so subtract 1 from line numbers)
    expected_lines = [4, 5]
    actual_lines = [v.loc.lineno for v in violations]
    assert actual_lines == expected_lines
