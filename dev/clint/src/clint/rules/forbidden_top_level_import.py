from clint.rules.base import Rule


class ForbiddenTopLevelImport(Rule):
    def __init__(self, module: str) -> None:
        self.module = module

    def _message(self) -> str:
        return (
            f"Importing module `{self.module}` at the top level is not allowed "
            "in this file. Use lazy import instead."
        )
