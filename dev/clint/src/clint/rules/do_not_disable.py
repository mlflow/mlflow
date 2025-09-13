from typing_extensions import Self

from clint.rules.base import Rule


class DoNotDisable(Rule):
    DO_NOT_DISABLE = {
        "B006": "Use None as default and set value in function body instead of mutable defaults"
    }

    def __init__(self, rules: set[str]) -> None:
        self.rules = rules

    @classmethod
    def check(cls, rules: set[str]) -> Self | None:
        if s := rules.intersection(DoNotDisable.DO_NOT_DISABLE.keys()):
            return cls(s)
        return None

    def _message(self) -> str:
        if len(self.rules) == 1:
            rule = next(iter(self.rules))
            hint = DoNotDisable.DO_NOT_DISABLE.get(rule)
            if hint:
                return f"DO NOT DISABLE {rule}: {hint}"
            else:
                return f"DO NOT DISABLE: {rule}"
        else:
            # Fallback for multiple rules (though currently only B006 is defined)
            hints = []
            for rule in sorted(self.rules):
                hint = DoNotDisable.DO_NOT_DISABLE.get(rule)
                if hint:
                    hints.append(f"{rule}: {hint}")
                else:
                    hints.append(rule)
            return f"DO NOT DISABLE: {', '.join(hints)}"
