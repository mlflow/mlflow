import pydantic

from mlflow.types import AgentInfo


class InputSchema(pydantic.BaseModel):
    query: str


class OutputSchema(pydantic.BaseModel):
    answer: str


def test_agent_info_to_dict_excludes_none_fields():
    info = AgentInfo(name="rag-assistant")

    assert info.to_dict() == {"name": "rag-assistant"}


def test_agent_info_to_dict_normalizes_pydantic_schema_classes():
    info = AgentInfo(
        metadata={
            "custom_inputs_schema": InputSchema,
            "custom_outputs_schema": OutputSchema,
        }
    )

    payload = info.to_dict()

    assert payload["metadata"]["custom_inputs_schema"] == InputSchema.model_json_schema()
    assert payload["metadata"]["custom_outputs_schema"] == OutputSchema.model_json_schema()


def test_agent_info_to_dict_normalizes_nested_schema_instances():
    info = AgentInfo(
        metadata={
            "custom_inputs_schema": InputSchema(query="hello"),
            "nested": {"custom_outputs_schema": OutputSchema},
        }
    )

    payload = info.to_dict()

    assert payload["metadata"]["custom_inputs_schema"] == InputSchema.model_json_schema()
    assert (
        payload["metadata"]["nested"]["custom_outputs_schema"]
        == OutputSchema.model_json_schema()
    )
