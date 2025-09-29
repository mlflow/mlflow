from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
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
    assert violations[0].loc == Location(3, 4)
    assert violations[1].loc == Location(8, 4)
    assert violations[2].loc == Location(13, 4)
