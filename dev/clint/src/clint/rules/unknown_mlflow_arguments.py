from clint.rules.base import Rule


class UnknownMlflowArguments(Rule):
    def __init__(self, function_name: str, unknown_args: set[str]) -> None:
        self.function_name = function_name
        self.unknown_args = unknown_args

    def _message(self) -> str:
        args_str = ", ".join(f"`{arg}`" for arg in sorted(self.unknown_args))
        return (
            f"Unknown arguments {args_str} passed to `{self.function_name}`. "
            "Check the function signature for valid parameter names."
        )
