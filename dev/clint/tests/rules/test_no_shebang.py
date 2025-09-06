from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules import NoShebang


def test_no_shebang(index_path: Path, tmp_path: Path) -> None:
    """Test that shebang lines are detected in Python files."""
    tmp_file = tmp_path / "test.py"

    # Test file with shebang - should trigger violation
    tmp_file.write_text(
        """#!/usr/bin/env python
import os

def hello():
    print("Hello, world!")
"""
    )
    config = Config(select={NoShebang.name})
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, NoShebang)
    assert results[0].loc == Location(0, 0)  # First line, first column (0-indexed)

    # Test file without shebang - should not trigger violation
    tmp_file.write_text(
        """import os

def hello():
    print("Hello, world!")
"""
    )
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 0


def test_no_shebang_various_patterns(index_path: Path, tmp_path: Path) -> None:
    """Test various shebang patterns."""
    tmp_file = tmp_path / "test.py"
    config = Config(select={NoShebang.name})

    # Test different shebang patterns
    shebang_patterns = [
        "#!/usr/bin/env python",
        "#!/usr/bin/python",
        "#!/usr/bin/python3",
        "#!/usr/bin/env python3",
        "#! /usr/bin/env python",  # With space after #!
    ]

    for pattern in shebang_patterns:
        tmp_file.write_text(f"{pattern}\nprint('hello')\n")
        results = lint_file(tmp_file, config, index_path)
        assert len(results) == 1, f"Failed to detect shebang: {pattern}"
        assert isinstance(results[0].rule, NoShebang)
        assert results[0].loc == Location(0, 0)


def test_no_shebang_edge_cases(index_path: Path, tmp_path: Path) -> None:
    """Test edge cases for shebang detection."""
    tmp_file = tmp_path / "test.py"
    config = Config(select={NoShebang.name})

    # Empty file - should not trigger
    tmp_file.write_text("")
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 0

    # Whitespace only - should not trigger
    tmp_file.write_text("   \n   \n")
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 0

    # Shebang not on first line - should not trigger (not a valid shebang)
    tmp_file.write_text("""
#!/usr/bin/env python
print("hello")
""")
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 0

    # Comment that starts with # but not shebang - should not trigger
    tmp_file.write_text("""# This is a comment
import os
""")
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 0
