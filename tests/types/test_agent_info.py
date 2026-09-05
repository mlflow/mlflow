import pytest

from mlflow.types.agent_info import AgentInfo
from mlflow.version import VERSION


def test_default_fields(monkeypatch):
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    info = AgentInfo()
    assert info.name == "mlflow_agent_server"
    assert info.use_case == "agent"
    assert info.mlflow_version == VERSION
    assert info.agent_api is None
    assert info.description is None
    assert info.version is None
    assert info.metadata is None


def test_name_from_databricks_app_name_env_var(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "from-env")
    info = AgentInfo()
    assert info.name == "from-env"


def test_explicit_name_overrides_env_var(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "from-env")
    info = AgentInfo(name="explicit")
    assert info.name == "explicit"


def test_model_dump_excludes_none(monkeypatch):
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    info = AgentInfo()
    result = info.model_dump(exclude_none=True)
    assert result == {
        "name": "mlflow_agent_server",
        "use_case": "agent",
        "mlflow_version": VERSION,
    }
    assert "agent_api" not in result
    assert "description" not in result
    assert "version" not in result
    assert "metadata" not in result


def test_all_fields_populated():
    info = AgentInfo(
        name="my-agent",
        use_case="agent",
        agent_api="responses",
        description="A test agent",
        version="1.0.0",
        metadata={
            "custom_inputs_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            "custom_outputs_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
            "team": "ml",
        },
    )
    result = info.model_dump(exclude_none=True)
    assert result["name"] == "my-agent"
    assert result["use_case"] == "agent"
    assert result["mlflow_version"] == VERSION
    assert result["agent_api"] == "responses"
    assert result["description"] == "A test agent"
    assert result["version"] == "1.0.0"
    assert result["metadata"]["custom_inputs_schema"]["properties"]["query"]["type"] == "string"
    assert result["metadata"]["custom_outputs_schema"]["properties"]["answer"]["type"] == "string"
    assert result["metadata"]["team"] == "ml"


def test_metadata_with_schemas():
    info = AgentInfo(
        name="schema-agent",
        metadata={
            "custom_inputs_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "temperature": {"type": "number"}},
            },
            "custom_outputs_schema": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
        },
    )
    result = info.model_dump(exclude_none=True)
    assert "custom_inputs_schema" in result["metadata"]
    assert "custom_outputs_schema" in result["metadata"]
    assert "query" in result["metadata"]["custom_inputs_schema"]["properties"]


def test_custom_name_overrides_default():
    info = AgentInfo(name="custom-name")
    assert info.name == "custom-name"
    result = info.model_dump(exclude_none=True)
    assert result["name"] == "custom-name"


def test_subclassing():
    class ExtendedAgentInfo(AgentInfo):
        capabilities: list[str] = []

    info = ExtendedAgentInfo(name="extended", capabilities=["qa", "search"])
    result = info.model_dump(exclude_none=True)
    assert result["name"] == "extended"
    assert result["capabilities"] == ["qa", "search"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("description", "A helpful agent"),
        ("version", "2.0.0"),
        ("metadata", {"team": "backend"}),
    ],
)
def test_optional_fields_individually(field, value):
    info = AgentInfo(**{field: value})
    result = info.model_dump(exclude_none=True)
    assert result[field] == value
