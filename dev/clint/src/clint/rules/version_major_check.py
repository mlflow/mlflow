import ast
import re
from typing import TYPE_CHECKING

from clint.rules.base import Rule

if TYPE_CHECKING:
    from clint.resolver import Resolver


class MajorVersionCheck(Rule):
    def _message(self) -> str:
        return (
            "Use `.major` field for major version comparisons instead of full version strings. "
            "This is more explicit, and efficient (avoids creating a second Version object). "
            "For example, use `Version(__version__).major >= 1` instead of "
            '`Version(__version__) >= Version("1.0.0")`.'
        )

    @staticmethod
    def check(node: ast.Compare, resolver: "Resolver") -> bool:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return False

        if not isinstance(node.ops[0], (ast.GtE, ast.LtE, ast.Gt, ast.Lt, ast.Eq, ast.NotEq)):
            return False

        if not (
            isinstance(node.left, ast.Call)
            and MajorVersionCheck._is_version_call(node.left, resolver)
        ):
            return False

        comparator = node.comparators[0]
        if not (
            isinstance(comparator, ast.Call)
            and MajorVersionCheck._is_version_call(comparator, resolver)
        ):
            return False

        match comparator.args:
            case [arg] if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                version_str = arg.value
                return MajorVersionCheck._is_major_only_version(version_str)

        return False

    @staticmethod
    def _is_version_call(node: ast.Call, resolver: "Resolver") -> bool:
        if resolved := resolver.resolve(node.func):
            return resolved == ["packaging", "version", "Version"]
        return False

    @staticmethod
    def _is_major_only_version(version_str: str) -> bool:
        pattern = r"^(\d+)\.0\.0$"
        return re.match(pattern, version_str) is not None
