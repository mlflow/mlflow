"""Secret scope definitions for MLflow secrets management."""

from enum import IntEnum


class SecretScope(IntEnum):
    """
    Enum defining the scope of secrets in MLflow.

    Attributes:
        GLOBAL: Global scope, accessible across all experiments (value: 0).
        SCORER: Scorer-specific scope, accessible only within a specific scorer (value: 1).
    """

    GLOBAL = 0
    SCORER = 1

    @classmethod
    def from_string(cls, scope: str) -> "SecretScope":
        """
        Convert a string to a SecretScope enum value (case-insensitive).

        Args:
            scope: String representation of the scope (e.g., 'global', 'SCORER', 'Scorer').

        Returns:
            SecretScope: The corresponding SecretScope enum value.

        Raises:
            ValueError: If the scope string doesn't match any valid scope.

        Example:
            >>> SecretScope.from_string("global")
            <SecretScope.GLOBAL: 0>
            >>> SecretScope.from_string("SCORER")
            <SecretScope.SCORER: 1>
        """
        try:
            return cls[scope.upper()]
        except KeyError:
            valid_scopes = ", ".join([s.name for s in cls])
            raise ValueError(f"Invalid scope '{scope}'. Valid scopes are: {valid_scopes}")
