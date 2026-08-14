import ast

from clint.rules.base import Rule


class ImplicitOptional(Rule):
    def _message(self) -> str:
        return "Use `Optional` if default value is `None`"

    @staticmethod
    def check(node: ast.AnnAssign) -> bool:
        """
        Returns True if the value to assign is `None` but the type annotation is
        not `Optional[...]` or `... | None`. For example: `a: int = None`.
        """
        if not ImplicitOptional._is_none(node.value):
            return False

        # Parse stringified annotations
        if isinstance(node.annotation, ast.Constant) and isinstance(node.annotation.value, str):
            try:
                parsed = ast.parse(node.annotation.value, mode="eval")
                ann = parsed.body
            except (SyntaxError, ValueError):
                # If parsing fails, the annotation is invalid and we trigger the rule
                # since we cannot verify it contains Optional or | None
                return True
        else:
            ann = node.annotation

        return not (ImplicitOptional._is_optional(ann) or ImplicitOptional._is_bitor_none(ann))

    @staticmethod
    def _is_optional(ann: ast.expr) -> bool:
        """
        Returns True if `ann` looks like `Optional[...]`.
        """
        return (
            isinstance(ann, ast.Subscript)
            and isinstance(ann.value, ast.Name)
            and ann.value.id == "Optional"
        )

    @staticmethod
    def _is_bitor_none(ann: ast.expr) -> bool:
        """
        Returns True if `ann` looks like `... | None`.
        """
        return (
            isinstance(ann, ast.BinOp)
            and isinstance(ann.op, ast.BitOr)
            and (isinstance(ann.right, ast.Constant) and ann.right.value is None)
        )

    @staticmethod
    def _is_none(value: ast.expr | None) -> bool:
        """
        Returns True if `value` represents `None`.
        """
        return isinstance(value, ast.Constant) and value.value is None
