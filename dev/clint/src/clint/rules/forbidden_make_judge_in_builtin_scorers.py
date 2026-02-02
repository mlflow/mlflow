import ast
from pathlib import Path

from clint.resolver import Resolver
from clint.rules.base import Rule


class ForbiddenMakeJudgeInBuiltinScorers(Rule):
    """Ensure make_judge is not used in builtin_scorers.py.

    After switching to InstructionsJudge in builtin_scorers.py, this rule
    prevents future regressions by detecting any usage of make_judge in that file.
    """

    def _message(self) -> str:
        return (
            "Usage of `make_judge` is forbidden in builtin_scorers.py. "
            "Use `InstructionsJudge` directly instead."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver, path: Path) -> bool:
        """Check if this is a call to make_judge in builtin_scorers.py.

        Args:
            node: The AST Call node to check
            resolver: Resolver instance to resolve fully qualified names
            path: Path to the file being linted

        Returns:
            True if this is a forbidden make_judge call, False otherwise
        """
        if path.name != "builtin_scorers.py":
            return False

        if names := resolver.resolve(node):
            match names:
                case ["mlflow", "genai", "judges", "make_judge", *_]:
                    return True
                case ["mlflow", "genai", "make_judge", *_]:
                    return True
                case ["make_judge", *_]:
                    return True
                case _:
                    return False
        return False
