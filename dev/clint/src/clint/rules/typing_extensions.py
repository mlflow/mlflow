from clint.rules.base import Rule


class TypingExtensions(Rule):
    def __init__(self, *, full_name: str, allowlist: list[str]) -> None:
        self.full_name = full_name
        self.allowlist = allowlist

    def _message(self) -> str:
        return (
            f"`{self.full_name}` is not allowed to use. Only {self.allowlist} are allowed. "
            "You can extend `tool.clint.typing-extensions-allowlist` in `pyproject.toml` if needed "
            "but make sure that the version requirement for `typing-extensions` is compatible with "
            "the added types."
        )
