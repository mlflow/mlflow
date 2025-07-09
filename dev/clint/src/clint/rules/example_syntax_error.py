from clint.rules.base import Rule


class ExampleSyntaxError(Rule):
    def _message(self) -> str:
        return "This example has a syntax error."
