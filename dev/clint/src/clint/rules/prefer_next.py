import ast

from clint.rules.base import Rule


class PreferNext(Rule):
    def _message(self) -> str:
        return (
            "Use `next(x for x in items if condition)` instead of "
            "`[x for x in items if condition][0]` for finding the first matching element."
        )

    @staticmethod
    def check(node: ast.Subscript) -> bool:
        """
        Returns True if the node is a list comprehension with an `if` clause
        subscripted with `[0]`.

        Examples that should be flagged:
        - [x for x in items if f(x)][0]
        - [x.name for x in items if x.active][0]

        Examples that should NOT be flagged:
        - [x for x in items][0]  (no if clause)
        - [x for x in items if f(x)][1]  (not [0])
        - [x for x in items if f(x)][-1]  (not [0])
        - (x for x in items if f(x))  (already a generator)
        """
        match node:
            case ast.Subscript(
                value=ast.ListComp(generators=generators),
                slice=ast.Constant(value=0),
            ) if any(gen.ifs for gen in generators):
                return True
            case _:
                return False
