from typing_extensions import Self

from clint.rules.base import Rule


class DoNotDisable(Rule):
    DO_NOT_DISABLE = {"B006"}

    def __init__(self, rules: set[str]) -> None:
        self.rules = rules

    @classmethod
    def check(cls, rules: set[str]) -> Self | None:
        if s := rules.intersection(DoNotDisable.DO_NOT_DISABLE):
            return cls(s)
        return None

    def _message(self) -> str:
        return f"DO NOT DISABLE: {self.rules}."
