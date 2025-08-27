import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class DeprecationWarningUsage(Rule):
    def _message(self) -> str:
        return (
            "Do not use `warnings.warn` with `DeprecationWarning` as Python does not show "
            "DeprecationWarning by default. Use `category=FutureWarning` instead."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if the call is warnings.warn() with DeprecationWarning as the category.

        Detects patterns like:
        - warnings.warn('message', DeprecationWarning)
        - warnings.warn('message', category=DeprecationWarning)
        - warnings.warn(category=DeprecationWarning, ...)
        - warn('message', DeprecationWarning) from 'from warnings import warn'
        """
        if names := resolver.resolve(node):
            # Check if this is a warnings.warn call or imported warn function
            if names == ["warnings", "warn"] or names == ["warn"]:
                # Check positional arguments - second argument could be DeprecationWarning
                if len(node.args) >= 2:
                    second_arg = node.args[1]
                    if isinstance(second_arg, ast.Name) and second_arg.id == "DeprecationWarning":
                        return True

                # Check keyword arguments for category=DeprecationWarning
                for keyword in node.keywords:
                    if (
                        keyword.arg == "category"
                        and isinstance(keyword.value, ast.Name)
                        and keyword.value.id == "DeprecationWarning"
                    ):
                        return True

        return False
