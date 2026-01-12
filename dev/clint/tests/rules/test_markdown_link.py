from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules.markdown_link import MarkdownLink


def test_markdown_link(index_path: Path) -> None:
    code = '''
# Bad
def function_with_markdown_link():
    """
    This function has a [markdown link](https://example.com).
    """

async def async_function_with_markdown_link():
    """
    This async function has a [markdown link](https://example.com).
    """

class MyClass:
    """
    Class with [another markdown link](https://test.com).
    """

# Good
def function_with_rest_link():
    """
    This function has a `reST link <https://example.com>`_.
    """
'''

    config = Config(select={MarkdownLink.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 3
    assert all(isinstance(v.rule, MarkdownLink) for v in violations)
    assert violations[0].range == Range(Position(3, 4))
    assert violations[1].range == Range(Position(8, 4))
    assert violations[2].range == Range(Position(13, 4))


def test_markdown_link_disable_on_end_line(index_path: Path) -> None:
    code = '''
def func():
    """
    Docstring with [markdown link](url).
    """  # clint: disable=markdown-link
    pass

async def async_func():
    """
    Async docstring with [markdown link](url).
    """  # clint: disable=markdown-link
    pass

class MyClass:
    """
    Class docstring with [markdown link](url).
    """  # clint: disable=markdown-link
    pass

# This should still be detected (no disable comment)
def func_without_disable():
    """
    Docstring with [markdown link](url).
    """
    pass
'''

    config = Config(select={MarkdownLink.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    # Only the last function without disable comment should have a violation
    assert len(violations) == 1
    assert isinstance(violations[0].rule, MarkdownLink)
