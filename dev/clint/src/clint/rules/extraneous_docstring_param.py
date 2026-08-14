from clint.rules.base import Rule


class ExtraneousDocstringParam(Rule):
    def __init__(self, params: set[str]) -> None:
        self.params = params

    def _message(self) -> str:
        return f"Extraneous parameters in docstring: {self.params}"
