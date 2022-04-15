from .pytest_raises_without_match import PytestRaisesWithoutMatch
from .print_function import PrintFunction
from .assert_raises_without_msg import AssertRaisesWithoutMsg


def register(linter):
    linter.register_checker(PytestRaisesWithoutMatch(linter))
    linter.register_checker(PrintFunction(linter))
    linter.register_checker(AssertRaisesWithoutMsg(linter))
