from clint.rules.base import Rule


class MissingNotebookH1Header(Rule):
    def _message(self) -> str:
        return "Notebook should have at least one H1 header for the title."
