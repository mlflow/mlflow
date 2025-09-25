import ast

from clint.rules.base import Rule


class IsinstanceUnionSyntax(Rule):
    def _message(self) -> str:
        return (
            "Use `isinstance(obj, (X, Y))` instead of `isinstance(obj, X | Y)` for better performance. "
            "The union syntax with `|` is slower than using a tuple of types."
        )

    @staticmethod
    def check(node: ast.Call) -> bool:
        """
        Returns True if the call is isinstance with union syntax (X | Y) in the second argument.

        Examples that should be flagged:
        - isinstance(obj, str | int)
        - isinstance(obj, int | str | float)
        - isinstance(value, (dict | list))

        Examples that should NOT be flagged:
        - isinstance(obj, (str, int))
        - isinstance(obj, str)
        - isinstance(obj, Union[str, int])
        - other_func(obj, str | int)
        """
        # Check if this is an isinstance call
        if not (isinstance(node.func, ast.Name) and node.func.id == "isinstance"):
            return False

        # isinstance must have at least 2 arguments
        if len(node.args) < 2:
            return False

        # Check if the second argument uses union syntax (X | Y)
        type_arg = node.args[1]
        return IsinstanceUnionSyntax._has_union_syntax(type_arg)

    @staticmethod
    def _has_union_syntax(node: ast.expr) -> bool:
        """
        Returns True if the node contains union syntax with BitOr operator.
        This handles nested cases like (A | B) | C.
        """
        match node:
            case ast.BinOp(op=ast.BitOr()):
                return True
            case ast.Tuple(elts=elements):
                # Check if any element in the tuple has union syntax
                return any(IsinstanceUnionSyntax._has_union_syntax(elt) for elt in elements)
            case _:
                return False
