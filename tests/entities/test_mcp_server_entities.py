import pytest

from mlflow.entities.mcp_access_binding import MCPAccessBinding
from mlflow.entities.mcp_server import (
    VALID_STATUS_TRANSITIONS,
    MCPRemoteTransportType,
    MCPServer,
    MCPStatus,
    MCPTool,
)
from mlflow.entities.mcp_server_version import MCPServerVersion
from mlflow.exceptions import MlflowException


def test_mcp_status_values():
    assert MCPStatus.DRAFT == "draft"
    assert MCPStatus.ACTIVE == "active"
    assert MCPStatus.DEPRECATED == "deprecated"
    assert MCPStatus.DELETED == "deleted"


def test_mcp_status_from_string():
    assert MCPStatus("draft") == MCPStatus.DRAFT
    assert MCPStatus("active") == MCPStatus.ACTIVE


def test_valid_status_transitions():
    assert VALID_STATUS_TRANSITIONS[MCPStatus.DRAFT] == {MCPStatus.ACTIVE, MCPStatus.DELETED}
    assert VALID_STATUS_TRANSITIONS[MCPStatus.ACTIVE] == {MCPStatus.DRAFT, MCPStatus.DEPRECATED}
    assert VALID_STATUS_TRANSITIONS[MCPStatus.DEPRECATED] == {MCPStatus.ACTIVE, MCPStatus.DELETED}
    assert MCPStatus.DELETED not in VALID_STATUS_TRANSITIONS


def test_mcp_remote_transport_type_values():
    assert MCPRemoteTransportType.STREAMABLE_HTTP == "streamable-http"
    assert MCPRemoteTransportType.SSE == "sse"


def test_mcp_tool_to_dict():
    tool = MCPTool(
        name="search",
        description="Search tool",
        input_schema={"type": "object"},
    )
    d = tool.to_dict()
    assert d == {"name": "search", "description": "Search tool", "inputSchema": {"type": "object"}}
    assert "outputSchema" not in d
    assert "title" not in d


def test_mcp_tool_from_dict():
    data = {"name": "search", "inputSchema": {"type": "object"}, "description": "Search"}
    tool = MCPTool.from_dict(data)
    assert tool.name == "search"
    assert tool.input_schema == {"type": "object"}
    assert tool.description == "Search"


def test_mcp_tool_from_dict_requires_name():
    with pytest.raises(MlflowException, match="Missing required key 'name'"):
        MCPTool.from_dict({"description": "missing name"})


def test_mcp_tool_roundtrip():
    original = MCPTool(
        name="search",
        title="Search",
        description="Search the web",
        input_schema={"type": "object"},
    )
    restored = MCPTool.from_dict(original.to_dict())
    assert original == restored


def test_mcp_server_workspace_resolution():
    server = MCPServer(name="test/server")
    assert server.workspace == "default"

    server2 = MCPServer(name="test/server", workspace="custom")
    assert server2.workspace == "custom"


def test_mcp_server_version_workspace_resolution():
    version = MCPServerVersion(
        name="test/server",
        version="1.0.0",
        server_json={"name": "test/server", "version": "1.0.0"},
    )
    assert version.workspace == "default"


def test_mcp_access_binding_workspace_resolution():
    binding = MCPAccessBinding(
        binding_id=1,
        server_name="test/server",
        endpoint_url="https://example.com/mcp",
    )
    assert binding.workspace == "default"
