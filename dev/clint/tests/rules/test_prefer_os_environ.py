from pathlib import Path

import pytest
from clint.config import Config
from clint.linter import lint_file
from clint.rules.prefer_os_environ import PreferOsEnviron


@pytest.mark.parametrize(
    "code",
    [
        pytest.param('import os\n\nval = os.getenv("FOO")', id="os.getenv"),
        pytest.param('import os\n\nval = os.getenv("FOO", "default")', id="os.getenv with default"),
        pytest.param('import os\n\nos.putenv("FOO", "bar")', id="os.putenv"),
        pytest.param('from os import getenv\n\nval = getenv("FOO")', id="from os import getenv"),
        pytest.param('from os import putenv\n\nputenv("FOO", "bar")', id="from os import putenv"),
    ],
)
def test_violation(code: str, index_path: Path) -> None:
    config = Config(select={PreferOsEnviron.name})
    violations = lint_file(Path("file.py"), code, config, index_path)
    assert len(violations) == 1
    assert isinstance(violations[0].rule, PreferOsEnviron)


@pytest.mark.parametrize(
    "code",
    [
        pytest.param('import os\n\nval = os.environ.get("FOO")', id="os.environ.get"),
        pytest.param('import os\n\nval = os.environ["FOO"]', id="os.environ subscript"),
        pytest.param('import os\n\nos.environ["FOO"] = "bar"', id="os.environ set"),
    ],
)
def test_no_violation(code: str, index_path: Path) -> None:
    config = Config(select={PreferOsEnviron.name})
    violations = lint_file(Path("file.py"), code, config, index_path)
    assert len(violations) == 0
