import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class InvalidAbstractMethod(Rule):
    def _message(self) -> str:
        return (
            "Abstract method should only contain a single statement/expression, "
            "and it must be `pass`, `...`, or a docstring."
        )

    @staticmethod
    def _is_abstract_method(
        node: ast.FunctionDef | ast.AsyncFunctionDef, resolver: Resolver
    ) -> bool:
        return any(
            (resolved := resolver.resolve(d)) and resolved == ["abc", "abstractmethod"]
            for d in node.decorator_list
        )

    @staticmethod
    def _has_invalid_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        # Does this abstract method have multiple statements/expressions?
        if len(node.body) > 1:
            return True

        # This abstract method has a single statement/expression.
        # Check if it's `pass`, `...`, or a docstring. If not, it's invalid.
        stmt = node.body[0]

        # Check for `pass`
        if isinstance(stmt, ast.Pass):
            return False

        # Check for `...` or docstring
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            value = stmt.value.value
            # `...` literal or docstring
            return not (value is ... or isinstance(value, str))

        # Any other statement is invalid
        return True

    @staticmethod
    def check(node: ast.FunctionDef | ast.AsyncFunctionDef, resolver: Resolver) -> bool:
        return InvalidAbstractMethod._is_abstract_method(
            node, resolver
        ) and InvalidAbstractMethod._has_invalid_body(node)
