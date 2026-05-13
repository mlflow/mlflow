from clint.rules.base import Rule


class UnknownMlflowFunction(Rule):
    def __init__(self, function_name: str) -> None:
        self.function_name = function_name

    def _message(self) -> str:
        return (
            f"Unknown MLflow function: `{self.function_name}`. "
            "This function may not exist or could be misspelled."
        )
