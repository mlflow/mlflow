from clint.rules.base import Rule


class NoRst(Rule):
    def _message(self) -> str:
        return "Do not use RST style. Use Google style instead."
