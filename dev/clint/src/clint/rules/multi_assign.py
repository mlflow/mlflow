import ast

from clint.rules.base import Rule


class MultiAssign(Rule):
    def _message(self) -> str:
        return (
            "Avoid multiple assignment (e.g., `x, y = 1, 2`). Use separate assignments "
            "instead for better readability and easier debugging."
        )

    @staticmethod
    def check(node: ast.Assign) -> bool:
        """
        Returns True if the assignment is a tuple assignment where the number of
        targets matches the number of values.

        Examples that should be flagged:
        - x, y = 1, 2
        - a, b, c = "test", "test2", "test3"
        - foo, bar = None, 1

        Examples that should NOT be flagged:
        - x, y = z
        - a, b = func()
        - x, y = get_coordinates()
        """
        # Check if we have exactly one target and it's a Tuple
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
            return False

        # Check if the value is also a Tuple
        if not isinstance(node.value, ast.Tuple):
            return False

        # Get the number of targets and values
        num_targets = len(node.targets[0].elts)
        num_values = len(node.value.elts)

        # Flag only when we have matching number of targets and values (at least 2)
        return num_targets == num_values and num_targets >= 2
