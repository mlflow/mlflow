from dataclasses import dataclass

from mlflow.entities.mcp_server import validate_mcp_server_name
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental


@experimental(version="3.15.0")
@dataclass(frozen=True)
class MCPServerRef:
    """
    An MCP server identity to record on a span.

    Args:
        name: Registered server name, such as ``com.example/weather``.
        version: Optional concrete publisher version. Mutually exclusive with ``alias``.
        alias: Optional alias to resolve when linking the span. Only the resolved concrete
            version is recorded. Without a version or alias, the version remains unknown.
    """

    name: str
    version: str | None = None
    alias: str | None = None

    def __post_init__(self) -> None:
        validate_mcp_server_name(self.name)
        if self.version is not None and self.alias is not None:
            raise MlflowException.invalid_parameter_value(
                "MCP server version and alias are mutually exclusive."
            )
        for field, value in (("version", self.version), ("alias", self.alias)):
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 256
                or "\x00" in value
                or any(0xD800 <= ord(char) <= 0xDFFF for char in value)
            ):
                raise MlflowException.invalid_parameter_value(
                    f"MCP server {field} must be a non-empty string of at most 256 characters "
                    "without NUL characters or surrogates."
                )
