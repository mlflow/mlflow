from clint.rules.base import Rule


class MarkdownLink(Rule):
    def _message(self) -> str:
        return (
            "Markdown link is not supported in docstring. "
            "Use reST link instead (e.g., `Link text <link URL>`_)."
        )
