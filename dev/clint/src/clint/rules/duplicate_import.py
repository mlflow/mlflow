from clint.rules.base import Rule


class DuplicateImport(Rule):
    def __init__(self, import_name: str) -> None:
        self.import_name = import_name

    def _message(self) -> str:
        return f"`{self.import_name}` is already imported. Consider removing it."
