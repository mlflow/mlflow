from .pytest_raises_checker import PytestRaisesChecker
from .print_function import PrintFunction
from .unittest_assert_raises import UnittestAssertRaises
from .string_checker import StringChecker
from .set_checker import SetChecker


def register(linter):
    linter.register_checker(PytestRaisesChecker(linter))
    linter.register_checker(PrintFunction(linter))
    linter.register_checker(UnittestAssertRaises(linter))
    linter.register_checker(SetChecker(linter))
    linter.register_checker(StringChecker(linter))
