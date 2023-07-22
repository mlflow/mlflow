from .assign_checker import AssignChecker
from .import_checker import ImportChecker
from .pytest_raises_checker import PytestRaisesChecker
from .string_checker import StringChecker
from .unittest_assert_raises import UnittestAssertRaises


def register(linter):
    linter.register_checker(PytestRaisesChecker(linter))
    linter.register_checker(UnittestAssertRaises(linter))
    linter.register_checker(StringChecker(linter))
    linter.register_checker(ImportChecker(linter))
    linter.register_checker(AssignChecker(linter))
