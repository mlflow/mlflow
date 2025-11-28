from clint.rules.base import Rule


class DuplicateImport(Rule):
    def __init__(self, name: str) -> None:
        self.import_name = name

    def _message(self) -> str:
        return f"`{self.import_name}` is already imported. Consider removing it."
