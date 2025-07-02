import ast

from clint.resolver import Resolver
from clint.rules.base import Rule


class TrackApiUsageOutermost(Rule):
    def _message(self) -> str:
        return (
            "The `@track_api_usage` decorator must be applied as the outermost decorator "
            "(first in the decorator list)."
        )

    @staticmethod
    def check(
        node: ast.expr,
        resolver: Resolver,
        decorator_list: list[ast.expr],
    ) -> bool:
        """
        Returns True if the `@track_api_usage` decorator from mlflow.telemetry.track is not
        used as the outermost decorator.

        Args:
            node: The decorator node being checked
            resolver: The resolver to identify the decorator
            decorator_list: The list of decorators for the node
        """
        resolved = resolver.resolve(node)
        if not resolved:
            return False

        if resolved != ["mlflow", "telemetry", "track", "track_api_usage"]:
            return False

        # Check if this decorator is not the outermost (first) in the decorator list
        for i, decorator in enumerate(decorator_list):
            if decorator is node:
                # If it's not at position 0 (outermost), it's a violation
                return i != 0

        return False
