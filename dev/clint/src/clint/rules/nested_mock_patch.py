import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class NestedMockPatch(Rule):
    def _message(self) -> str:
        return (
            "Do not nest `unittest.mock.patch` context managers. "
            "Use multiple context managers in a single `with` statement instead: "
            "`with mock.patch(...), mock.patch(...): ...`"
        )

    @staticmethod
    def check(node: ast.With, resolver: Resolver) -> bool:
        """
        Returns True if the with statement uses mock.patch and contains only a single
        nested with statement that also uses mock.patch.
        """
        # Check if the outer with statement uses mock.patch
        outer_has_mock_patch = any(
            NestedMockPatch._is_mock_patch(item.context_expr, resolver) for item in node.items
        )

        if not outer_has_mock_patch:
            return False

        # Check if the body has exactly one statement and it's a with statement
        if len(node.body) == 1 and isinstance(node.body[0], ast.With):
            # Check if the nested with statement also uses mock.patch
            inner_has_mock_patch = any(
                NestedMockPatch._is_mock_patch(item.context_expr, resolver)
                for item in node.body[0].items
            )

            if inner_has_mock_patch:
                return True

        return False

    @staticmethod
    def _is_mock_patch(node: ast.expr, resolver: Resolver) -> bool:
        """
        Returns True if the node is a call to mock.patch or any of its variants.
        """
        # Handle direct calls: mock.patch(...), mock.patch.object(...), etc.
        if isinstance(node, ast.Call):
            if res := resolver.resolve(node.func):
                match res:
                    # Matches unittest.mock.patch, unittest.mock.patch.object, etc.
                    case ["unittest", "mock", "patch", *_]:
                        return True
        return False
