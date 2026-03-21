from clint.rules.base import Rule


class TestNameTypo(Rule):
    def _message(self) -> str:
        return "This function looks like a test, but its name does not start with 'test_'."
