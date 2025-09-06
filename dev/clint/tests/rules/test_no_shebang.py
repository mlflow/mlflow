from pathlib import Path

import pytest
from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules import NoShebang


def test_no_shebang(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"

    # Test file with shebang - should trigger violation
    tmp_file.write_text(
        """#!/usr/bin/env python
print("Hello, world!")
"""
    )
    config = Config(select={NoShebang.name})
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, NoShebang)
    assert results[0].loc == Location(0, 0)  # First line, first column (0-indexed)

    # Test file without shebang - should not trigger violation
    tmp_file.write_text("""print("Hello, world!")""")
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 0


@pytest.mark.parametrize(
    "shebang_pattern",
    [
        "#!/usr/bin/env python",
        "#!/usr/bin/python",
        "#!/usr/bin/python3",
        "#!/usr/bin/env python3",
        "#! /usr/bin/env python",  # With space after #!
    ],
)
def test_no_shebang_various_patterns(
    index_path: Path, tmp_path: Path, shebang_pattern: str
) -> None:
    tmp_file = tmp_path / "test.py"
    config = Config(select={NoShebang.name})

    tmp_file.write_text(f"{shebang_pattern}\nprint('hello')\n")
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 1, f"Failed to detect shebang: {shebang_pattern}"
    assert isinstance(results[0].rule, NoShebang)
    assert results[0].loc == Location(0, 0)


@pytest.mark.parametrize(
    ("content", "description"),
    [
        ("", "empty file"),
        ("   \n   \n", "whitespace only"),
        ('\n#!/usr/bin/env python\nprint("hello")\n', "shebang not on first line"),
        ("# This is a comment\nimport os\n", "comment that starts with # but not shebang"),
    ],
)
def test_no_shebang_edge_cases(
    index_path: Path, tmp_path: Path, content: str, description: str
) -> None:
    tmp_file = tmp_path / "test.py"
    config = Config(select={NoShebang.name})

    tmp_file.write_text(content)
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 0, f"Should not trigger violation for: {description}"
