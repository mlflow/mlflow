from clint.rules.base import Rule


class UnusedDisableComment(Rule):
    def __init__(self, rule_name: str) -> None:
        self.rule_name = rule_name

    def _message(self) -> str:
        return f"Unused disable comment for rule `{self.rule_name}`"
