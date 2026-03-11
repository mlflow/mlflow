from clint.rules.base import Rule


class LazyImport(Rule):
    def _message(self) -> str:
        return "This module must be imported at the top level."
