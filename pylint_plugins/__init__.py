from .pytest_raises_without_match import PytestRaisesWithoutMatch
from .print_function import PrintFunction


def register(linter):
    linter.register_checker(PytestRaisesWithoutMatch(linter))
    linter.register_checker(PrintFunction(linter))
