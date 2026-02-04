from pathlib import Path

import pytest
from clint.config import Config
from clint.linter import lint_file
from clint.rules.use_gh_token import UseGhToken


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            'import os\n\ntoken = os.getenv("GITHUB_TOKEN")',
            id="os.getenv",
        ),
        pytest.param(
            'import os\n\ntoken = os.environ.get("GITHUB_TOKEN")',
            id="os.environ.get",
        ),
        pytest.param(
            'import os\n\ntoken = os.getenv("GITHUB_TOKEN", "default")',
            id="os.getenv with default",
        ),
        pytest.param(
            'import os\n\ntoken = os.environ.get("GITHUB_TOKEN", None)',
            id="os.environ.get with default",
        ),
    ],
)
def test_violation(code: str, index_path: Path) -> None:
    config = Config(select={UseGhToken.name})
    violations = lint_file(Path("file.py"), code, config, index_path)
    assert len(violations) == 1
    assert isinstance(violations[0].rule, UseGhToken)


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            'import os\n\ntoken = os.getenv("GH_TOKEN")',
            id="os.getenv with GH_TOKEN",
        ),
        pytest.param(
            'import os\n\ntoken = os.environ.get("GH_TOKEN")',
            id="os.environ.get with GH_TOKEN",
        ),
        pytest.param(
            'import os\n\ntoken = os.getenv("OTHER_VAR")',
            id="os.getenv with other var",
        ),
    ],
)
def test_no_violation(code: str, index_path: Path) -> None:
    config = Config(select={UseGhToken.name})
    violations = lint_file(Path("file.py"), code, config, index_path)
    assert len(violations) == 0
