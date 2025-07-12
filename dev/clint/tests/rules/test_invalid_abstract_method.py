from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules.invalid_abstract_method import InvalidAbstractMethod


def test_invalid_abstract_method(index: SymbolIndex, config: Config, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
import abc

class AbstractExample(abc.ABC):
    @abc.abstractmethod
    def good_abstract_method_pass(self) -> None:
        pass

    @abc.abstractmethod
    def good_abstract_method_ellipsis(self) -> None:
        ...

    @abc.abstractmethod
    def good_abstract_method_docstring(self) -> None:
        '''This is a valid docstring'''

    @abc.abstractmethod
    def bad_abstract_method_implemented(self) -> None:
        return "This should not be here"

    @abc.abstractmethod
    def bad_abstract_method_mulitple_statements(self) -> None:
        pass
        ...
"""
    )

    violations = lint_file(tmp_file, config, index)
    assert len(violations) == 2
    assert all(isinstance(v.rule, InvalidAbstractMethod) for v in violations)
    assert violations[0].loc == Location(17, 4)
    assert violations[1].loc == Location(21, 4)
