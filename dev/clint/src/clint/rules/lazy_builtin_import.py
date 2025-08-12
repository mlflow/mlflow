from clint.rules.base import Rule


class LazyBuiltinImport(Rule):
    def _message(self) -> str:
        return "Builtin modules must be imported at the top level."
