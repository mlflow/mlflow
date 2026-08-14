from clint.rules.base import Rule


class DocstringParamOrder(Rule):
    def __init__(self, params: list[str]) -> None:
        self.params = params

    def _message(self) -> str:
        return f"Unordered parameters in docstring: {self.params}"
