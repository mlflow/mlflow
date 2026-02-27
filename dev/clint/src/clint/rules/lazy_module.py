from clint.rules.base import Rule


class LazyModule(Rule):
    def _message(self) -> str:
        return "Module loaded by `LazyLoader` must be imported in `TYPE_CHECKING` block."
