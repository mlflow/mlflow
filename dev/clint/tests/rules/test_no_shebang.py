from pathlib import Path

import pytest
from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules import NoShebang


def test_no_shebang(index_path: Path) -> None:
    config = Config(select={NoShebang.name})

    # Test file with shebang - should trigger violation
    code = "#!/usr/bin/env python\nprint('hello')"
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert all(isinstance(r.rule, NoShebang) for r in results)
    assert results[0].loc == Location(0, 0)  # First line, first column (0-indexed)

    # Test file without shebang - should not trigger violation
    code = "print('hello')"
    results = lint_file(Path("test.py"), code, config, index_path)
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
def test_no_shebang_various_patterns(index_path: Path, shebang: str) -> None:
    config = Config(select={NoShebang.name})

    code = f"{shebang}\nprint('hello')\n"
    results = lint_file(Path("test.py"), code, config, index_path)
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
def test_no_shebang_edge_cases(index_path: Path, content: str) -> None:
    config = Config(select={NoShebang.name})

    code = content
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0
