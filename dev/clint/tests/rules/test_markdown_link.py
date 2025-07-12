from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import lint_file
from clint.rules.markdown_link import MarkdownLink


def test_markdown_link(index: SymbolIndex, config: Config, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        '''
def function_with_markdown_link():
    """
    This function has a [markdown link](https://example.com).
    """

def function_with_rest_link():
    """
    This function has a `reST link <https://example.com>`_.
    """

class MyClass:
    """
    Class with [another markdown link](https://test.com).
    """
'''
    )

    violations = lint_file(tmp_file, config, index)
    assert len(violations) == 2
    assert all(isinstance(v.rule, MarkdownLink) for v in violations)
    assert violations[0].loc.lineno == 2  # Function docstring
    assert violations[1].loc.lineno == 12  # Class docstring
