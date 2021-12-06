from .pytest_raises_without_match import PytestRaisesWithoutMatch


def register(linter):
    linter.register_checker(PytestRaisesWithoutMatch(linter))
