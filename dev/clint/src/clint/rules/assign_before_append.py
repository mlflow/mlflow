import ast

from clint.rules.base import Rule


class AssignBeforeAppend(Rule):
    def _message(self) -> str:
        return (
            "Avoid unnecessary assignment before appending to a list. "
            "Use a list comprehension instead."
        )

    @staticmethod
    def check(node: ast.For, prev_stmt: ast.stmt | None) -> bool:
        """
        Returns True if the for loop contains exactly two statements:
        an assignment followed by appending that variable to a list, AND
        the loop is immediately preceded by an empty list initialization.

        Examples that should be flagged:
        ---
        items = []
        for x in data:
            item = transform(x)
            items.append(item)
        ---
        """
        # Match: for loop with exactly 2 statements in body
        match node:
            case ast.For(body=[stmt1, stmt2]):
                pass
            case _:
                return False

        # Match stmt1: simple assignment (item = x)
        match stmt1:
            case ast.Assign(targets=[ast.Name(id=assigned_var)]):
                pass
            case _:
                return False

        # Match stmt2: list.append(item)
        match stmt2:
            case ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=list_name), attr="append"),
                    args=[ast.Name(id=appended_var)],
                )
            ):
                # Check if the appended variable is the same as the assigned variable
                if appended_var != assigned_var:
                    return False
            case _:
                return False

        # Only flag if prev_stmt is empty list initialization for the same list
        match prev_stmt:
            case ast.Assign(
                targets=[ast.Name(id=prev_list_name)],
                value=ast.List(elts=[]),
            ) if prev_list_name == list_name:
                return True
            case _:
                return False
