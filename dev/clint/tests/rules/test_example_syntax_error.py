from pathlib import Path

import pytest
from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.example_syntax_error import ExampleSyntaxError


def test_example_syntax_error(index_path: Path) -> None:
    code = '''
def bad():
    """
    .. code-block:: python

        def f():

    """

def good():
    """
    .. code-block:: python

        def f():
            return "This is a good example"
    """
'''
    config = Config(select={ExampleSyntaxError.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, ExampleSyntaxError) for v in violations)
    assert violations[0].loc == Location(5, 8)


@pytest.mark.parametrize("suffix", [".md", ".mdx"])
def test_example_syntax_error_markdown(index_path: Path, tmp_path: Path, suffix: str) -> None:
    tmp_file = (tmp_path / "test").with_suffix(suffix)
    code = """
```python
def g():
```
"""
    tmp_file.write_text(code)
    config = Config(select={ExampleSyntaxError.name})
    violations = lint_file(tmp_file, code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, ExampleSyntaxError) for v in violations)
    assert violations[0].loc == Location(2, 0)
