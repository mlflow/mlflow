import pytest
from pydantic import BaseModel

from mlflow.tracing.constant import SpanAttributeKey
from mlflow.types.agent_attribute import AgentAttribute, _normalize_schema


class SampleInputs(BaseModel):
    strategy: str
    max_depth: int = 5


class SampleOutputs(BaseModel):
    result: str
    confidence: float


def test_default_fields_are_none():
    attr = AgentAttribute()
    assert attr.name is None
    assert attr.description is None
    assert attr.version is None
    assert attr.custom_inputs_schema is None
    assert attr.custom_outputs_schema is None
    assert attr.tags is None


def test_to_dict_with_all_fields():
    attr = AgentAttribute(
        name="planner",
        description="Plans multi-step tasks",
        version="1.0",
        custom_inputs_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        custom_outputs_schema={"type": "object", "properties": {"y": {"type": "number"}}},
        tags={"team": "ml", "env": "prod"},
    )
    result = attr.to_dict()
    assert result["name"] == "planner"
    assert result["description"] == "Plans multi-step tasks"
    assert result["version"] == "1.0"
    assert result["custom_inputs_schema"] == {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }
    assert result["custom_outputs_schema"] == {
        "type": "object",
        "properties": {"y": {"type": "number"}},
    }
    assert result["tags"] == {"team": "ml", "env": "prod"}


def test_to_dict_omits_none():
    attr = AgentAttribute(name="planner")
    result = attr.to_dict()
    assert result == {"name": "planner"}
    assert "description" not in result
    assert "version" not in result
    assert "custom_inputs_schema" not in result
    assert "custom_outputs_schema" not in result
    assert "tags" not in result


@pytest.mark.parametrize(
    ("schema", "expected_type"),
    [
        (None, type(None)),
        ({"type": "object"}, dict),
        (SampleInputs, dict),
    ],
)
def test_normalize_schema(schema, expected_type):
    result = _normalize_schema(schema)
    if schema is None:
        assert result is None
    else:
        assert isinstance(result, expected_type)


def test_normalize_schema_dict_passthrough():
    raw = {"type": "object", "properties": {"x": {"type": "string"}}}
    assert _normalize_schema(raw) is raw


def test_normalize_schema_pydantic():
    result = _normalize_schema(SampleInputs)
    assert result == SampleInputs.model_json_schema()
    assert "properties" in result
    assert "strategy" in result["properties"]
    assert "max_depth" in result["properties"]


def test_normalize_schema_invalid_type():
    with pytest.raises(TypeError, match="Schema must be a dict"):
        _normalize_schema("not a schema")


def test_to_dict_with_pydantic_schemas():
    attr = AgentAttribute(
        name="planner",
        custom_inputs_schema=SampleInputs,
        custom_outputs_schema=SampleOutputs,
    )
    result = attr.to_dict()
    assert result["custom_inputs_schema"] == SampleInputs.model_json_schema()
    assert result["custom_outputs_schema"] == SampleOutputs.model_json_schema()


def test_get_span_attributes():
    attr = AgentAttribute(name="planner", description="A planner agent", version="2.0")
    span_attrs = attr.get_span_attributes()
    assert span_attrs == {
        SpanAttributeKey.AGENT_NAME: "planner",
        SpanAttributeKey.AGENT_DESCRIPTION: "A planner agent",
        SpanAttributeKey.AGENT_VERSION: "2.0",
    }


def test_get_span_attributes_omits_none():
    attr = AgentAttribute(name="planner")
    span_attrs = attr.get_span_attributes()
    assert span_attrs == {SpanAttributeKey.AGENT_NAME: "planner"}
    assert SpanAttributeKey.AGENT_DESCRIPTION not in span_attrs
    assert SpanAttributeKey.AGENT_VERSION not in span_attrs


def test_get_span_attributes_empty_when_all_none():
    attr = AgentAttribute()
    assert attr.get_span_attributes() == {}
