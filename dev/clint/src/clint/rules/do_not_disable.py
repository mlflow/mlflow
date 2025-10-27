from typing_extensions import Self

from clint.rules.base import Rule


class DoNotDisable(Rule):
    RULES = {
        "B006": "Use None as default and set value in function body instead of mutable defaults",
        "F821": "Use typing.TYPE_CHECKING for forward references to optional dependencies",
    }

    def __init__(self, rules: set[str]) -> None:
        self.rules = rules

    @classmethod
    def check(cls, rules: set[str]) -> Self | None:
        if s := rules.intersection(DoNotDisable.RULES.keys()):
            return cls(s)
        return None

    def _message(self) -> str:
        # Build message for all rules (works for single and multiple rules)
        hints = []
        for rule in sorted(self.rules):
            hint = DoNotDisable.RULES.get(rule)
            if hint:
                hints.append(f"{rule}: {hint}")
            else:
                hints.append(rule)
        return f"DO NOT DISABLE {', '.join(hints)}"
