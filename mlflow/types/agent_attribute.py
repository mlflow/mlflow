from dataclasses import dataclass
from typing import Any


def _normalize_schema(schema: Any) -> dict[str, Any] | None:
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    raise TypeError(
        f"Schema must be a dict (JSON Schema) or a Pydantic BaseModel class, got {type(schema)}"
    )


@dataclass
class AgentAttribute:
    """Optional metadata describing a ResponsesAgent's identity and interface.

    Args:
        name: Agent name used for discovery and span labeling.
            Defaults to the class name if not provided.
        description: Human-readable description of what the agent does.
        version: Version string for the agent.
        custom_inputs_schema: JSON Schema dict or Pydantic BaseModel class
            describing accepted custom inputs.
        custom_outputs_schema: JSON Schema dict or Pydantic BaseModel class
            describing produced custom outputs.
        tags: Arbitrary key-value metadata for extensibility.
    """

    name: str | None = None
    description: str | None = None
    version: str | None = None
    custom_inputs_schema: dict[str, Any] | type | None = None
    custom_outputs_schema: dict[str, Any] | type | None = None
    tags: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.name is not None:
            result["name"] = self.name
        if self.description is not None:
            result["description"] = self.description
        if self.version is not None:
            result["version"] = self.version

        normalized_inputs = _normalize_schema(self.custom_inputs_schema)
        if normalized_inputs is not None:
            result["custom_inputs_schema"] = normalized_inputs

        normalized_outputs = _normalize_schema(self.custom_outputs_schema)
        if normalized_outputs is not None:
            result["custom_outputs_schema"] = normalized_outputs

        if self.tags is not None:
            result["tags"] = self.tags
        return result

    def get_span_attributes(self) -> dict[str, str]:
        from mlflow.tracing.constant import SpanAttributeKey

        attrs: dict[str, str] = {}
        if self.name is not None:
            attrs[SpanAttributeKey.AGENT_NAME] = self.name
        if self.description is not None:
            attrs[SpanAttributeKey.AGENT_DESCRIPTION] = self.description
        if self.version is not None:
            attrs[SpanAttributeKey.AGENT_VERSION] = self.version
        return attrs
