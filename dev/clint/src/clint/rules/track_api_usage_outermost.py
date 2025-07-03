import ast
from typing import Optional

from clint.resolver import Resolver
from clint.rules.base import Rule


class TrackApiUsageOutermost(Rule):
    def _message(self) -> str:
        return (
            "The `@track_api_usage` decorator must be applied as the topmost decorator "
            "to ensure proper telemetry tracking. It needs to wrap the complete decorated "
            "function to accurately check if the API is invoked from internal MLflow code."
        )

    @staticmethod
    def check(
        resolver: Resolver,
        decorator_list: list[ast.expr],
    ) -> Optional[ast.expr]:
        """
        Returns the decorator node if it is not the outermost decorator.

        Args:
            resolver: The resolver to identify the decorator
            decorator_list: The list of decorators for the node
        """

        for i, decorator in enumerate(decorator_list):
            resolved = resolver.resolve(decorator)
            # If it's not at position 0 (outermost), it's a violation
            if resolved == ["mlflow", "telemetry", "track", "track_api_usage"] and i != 0:
                return decorator
