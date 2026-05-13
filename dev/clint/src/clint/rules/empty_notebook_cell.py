from clint.rules.base import Rule


class EmptyNotebookCell(Rule):
    def _message(self) -> str:
        return "Empty notebook cell. Remove it or add some content."
