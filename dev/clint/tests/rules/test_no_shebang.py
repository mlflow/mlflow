from pathlib import Path

import pytest
from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules import NoShebang


def test_no_shebang(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"

    # Test file with shebang - should trigger violation
    tmp_file.write_text("#!/usr/bin/env python\nprint('hello')")
    config = Config(select={NoShebang.name})
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 1
    assert all(isinstance(r.rule, NoShebang) for r in results)
    assert results[0].loc == Location(0, 0)  # First line, first column (0-indexed)

    # Test file without shebang - should not trigger violation
    tmp_file.write_text("print('hello')")
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 0


@pytest.mark.parametrize(
    "shebang",
    [
        "#!/usr/bin/env python",
        "#!/usr/bin/python",
        "#!/usr/bin/python3",
        "#!/usr/bin/env python3",
        "#! /usr/bin/env python",  # With space after #!
    ],
)
def test_no_shebang_various_patterns(index_path: Path, tmp_path: Path, shebang: str) -> None:
    tmp_file = tmp_path / "test.py"
    config = Config(select={NoShebang.name})

    tmp_file.write_text(f"{shebang}\nprint('hello')\n")
    results = lint_file(tmp_file, config, index_path)
    assert all(isinstance(r.rule, NoShebang) for r in results)
    assert results[0].loc == Location(0, 0)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "   \n   \n",
        '\n#!/usr/bin/env python\nprint("hello")\n',
        "# This is a comment\nimport os\n",
    ],
    ids=[
        "empty_file",
        "whitespace_only",
        "shebang_not_on_first_line",
        "comment_not_shebang",
    ],
)
def test_no_shebang_edge_cases(index_path: Path, tmp_path: Path, content: str) -> None:
    tmp_file = tmp_path / "test.py"
    config = Config(select={NoShebang.name})

    tmp_file.write_text(content)
    results = lint_file(tmp_file, config, index_path)
    assert len(results) == 0
