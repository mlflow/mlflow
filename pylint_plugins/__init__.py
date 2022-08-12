from .pytest_raises_checker import PytestRaisesChecker
from .print_function import PrintFunction
from .unittest_assert_raises import UnittestAssertRaises


def register(linter):
    linter.register_checker(PytestRaisesChecker(linter))
    linter.register_checker(PrintFunction(linter))
    linter.register_checker(UnittestAssertRaises(linter))
