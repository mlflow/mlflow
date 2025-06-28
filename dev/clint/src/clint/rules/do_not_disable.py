from clint.rules.base import Rule


class DoNotDisable(Rule):
    DO_NOT_DISABLE = {"B006"}

    def __init__(self, rules: set[str]) -> None:
        self.rules = rules

    @classmethod
    def check(cls, rules: set[str]) -> "DoNotDisable":
        if s := rules.intersection(DoNotDisable.DO_NOT_DISABLE):
            return cls(s)

    def _message(self) -> str:
        return f"DO NOT DISABLE: {self.rules}."
