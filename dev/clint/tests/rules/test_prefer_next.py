from pathlib import Path

import pytest
from clint.config import Config
from clint.linter import lint_file
from clint.rules import PreferNext


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("[x for x in items if f(x)][0]", id="basic_pattern"),
    ],
)
def test_flag(index_path: Path, code: str) -> None:
    config = Config(select={PreferNext.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, PreferNext)


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("[x for x in items][0]", id="no_if_clause"),
        pytest.param("[x for x in items if f(x)][1]", id="not_zero_index"),
        pytest.param("[x for x in items if f(x)][-1]", id="negative_index"),
        pytest.param("(x for x in items if f(x))", id="already_generator"),
        pytest.param("next(x for x in items if f(x))", id="already_using_next"),
        pytest.param("[x for x in items if f(x)]", id="no_subscript"),
        pytest.param("items[0]", id="simple_subscript"),
    ],
)
def test_no_flag(index_path: Path, code: str) -> None:
    config = Config(select={PreferNext.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 0
