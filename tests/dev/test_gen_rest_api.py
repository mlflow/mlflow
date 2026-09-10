from fastapi import FastAPI
from pydantic import BaseModel, RootModel

from dev.gen_rest_api import _build_fastapi_endpoint_docs


def test_build_fastapi_endpoint_docs_handles_recursive_models():
    class Node(BaseModel):
        name: str
        children: list["Node"] = []

    app = FastAPI()

    @app.get("/tree", response_model=Node)
    def get_tree():
        return Node(name="root")

    endpoints, schemas = _build_fastapi_endpoint_docs(app)

    assert [field.name for field in endpoints[0].response_sections[0].fields] == [
        "name",
        "children",
    ]
    assert endpoints[0].response_sections[0].fields[1].field_type == (
        "An array of :ref:`fastapiNode`"
    )
    assert [section.title for section in schemas] == ["Node"]


def test_build_fastapi_endpoint_docs_handles_root_array_models():
    class Items(RootModel[list[str]]):
        pass

    app = FastAPI()

    @app.get("/items", response_model=Items)
    def get_items():
        return Items(["item"])

    endpoints, _ = _build_fastapi_endpoint_docs(app)

    assert [field.name for field in endpoints[0].response_sections[0].fields] == ["value"]
    assert endpoints[0].response_sections[0].fields[0].field_type == "An array of ``STRING``"


def test_build_fastapi_endpoint_docs_includes_gateway_and_mcp_routes():
    endpoints, _ = _build_fastapi_endpoint_docs()
    endpoint_map = {(endpoint.method, endpoint.path): endpoint for endpoint in endpoints}

    gateway_endpoint = endpoint_map[("POST", "/gateway/{endpoint_name}/mlflow/invocations")]
    assert [field.name for field in gateway_endpoint.request_sections[0].fields] == [
        "endpoint_name"
    ]
    openai_endpoint = endpoint_map[("POST", "/gateway/openai/v1/chat/completions")]
    assert openai_endpoint.title == "OpenAI Passthrough Chat"
    assert openai_endpoint.category == "Gateway APIs"
    otlp_endpoint = endpoint_map[("POST", "/v1/traces")]
    assert otlp_endpoint.title == "Export Traces"
    assert otlp_endpoint.category == "OpenTelemetry APIs"
    workspace_field = next(
        field
        for section in otlp_endpoint.request_sections
        for field in section.fields
        if field.name == "X-MLFLOW-WORKSPACE"
    )
    assert "Workspace to use when MLflow workspaces are enabled." in workspace_field.description
    otlp_description = otlp_endpoint.description
    assert "Args:" not in otlp_description
    assert "Returns:" not in otlp_description
    assert "Raises:" not in otlp_description

    mcp_endpoint = endpoint_map[("POST", "/api/3.0/mlflow/mcp-servers")]
    assert [field.name for field in mcp_endpoint.request_sections[0].fields] == [
        "name",
        "description",
        "icons",
    ]
    assert mcp_endpoint.category == "MCP Server Registry APIs"
    assert mcp_endpoint.response_sections


def test_build_fastapi_endpoint_docs_discovers_mounted_routes():
    class Response(BaseModel):
        value: str

    app = FastAPI()

    @app.get("/new-endpoint", response_model=Response)
    def new_endpoint():
        return Response(value="value")

    endpoints, _ = _build_fastapi_endpoint_docs(app)

    assert [(endpoint.method, endpoint.path) for endpoint in endpoints] == [
        ("GET", "/new-endpoint")
    ]
    assert [field.name for field in endpoints[0].response_sections[0].fields] == ["value"]
