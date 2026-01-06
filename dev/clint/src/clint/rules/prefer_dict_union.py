import ast

from clint.rules.base import Rule


def _is_simple_name_or_attribute(node: ast.expr) -> bool:
    """
    Check if a node is a simple name (e.g., `a`) or a chain of attribute
    accesses on a simple name (e.g., `obj.attr` or `a.b.c`).
    """
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _is_simple_name_or_attribute(node.value)
    return False


class PreferDictUnion(Rule):
    def _message(self) -> str:
        return (
            "Use `|` operator for dictionary merging (e.g., `a | b`) "
            "instead of `{**a, **b}` for better readability."
        )

    @staticmethod
    def check(node: ast.Dict) -> bool:
        """
        Returns True if the dictionary is composed entirely of 2+ dictionary unpacking
        expressions that can be replaced with the `|` operator.

        Examples that should be flagged:
        - {**a, **b}
        - {**a, **b, **c}
        - {**obj.attr, **b}
        - {**a.b.c, **d}

        Examples that should NOT be flagged:
        - {**a}  # Single unpack
        - {**a, "key": value}  # Mixed with literal keys
        - {**data[0], **b}, {**func(), **b}  # Complex expressions
        """
        # Need at least 2 elements for a merge
        if len(node.keys) < 2:
            return False

        # All keys must be None (indicating dictionary unpacking with **)
        if not all(key is None for key in node.keys):
            return False

        # All values must be simple names or attribute access on a name
        return all(_is_simple_name_or_attribute(value) for value in node.values)
