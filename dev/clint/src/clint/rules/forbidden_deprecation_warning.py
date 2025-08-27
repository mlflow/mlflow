import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class ForbiddenDeprecationWarning(Rule):
    def _message(self) -> str:
        return (
            "Do not use DeprecationWarning with warnings.warn(). "
            "Use FutureWarning instead since Python does not show DeprecationWarning by default."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if `node` looks like `warnings.warn(..., category=DeprecationWarning)`.
        """
        # Check if this is a call to warnings.warn
        if resolved := resolver.resolve(node.func):
            if resolved == ["warnings", "warn"]:
                # Check if there's a category keyword argument with DeprecationWarning
                for keyword in node.keywords:
                    if (
                        keyword.arg == "category"
                        and isinstance(keyword.value, ast.Name)
                        and keyword.value.id == "DeprecationWarning"
                    ):
                        return True

                # Check if DeprecationWarning is passed as the second positional argument (category)
                if (
                    len(node.args) >= 2
                    and isinstance(node.args[1], ast.Name)
                    and node.args[1].id == "DeprecationWarning"
                ):
                    return True
        return False
