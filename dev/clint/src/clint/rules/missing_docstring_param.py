from clint.rules.base import Rule


class MissingDocstringParam(Rule):
    def __init__(self, params: set[str]) -> None:
        self.params = params

    def _message(self) -> str:
        return f"Missing parameters in docstring: {self.params}"
