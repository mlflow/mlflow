from dataclasses import FrozenInstanceError

import pytest

from mlflow.entities import MCPServerRef
from mlflow.exceptions import MlflowException


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"version": "1.0.0+Build"},
        {"version": "1"},
        *[{"version": value} for value in [" 1.0.0", "1.0.0 ", "\t1.0.0", "1.0.0\n"]],
        {"alias": "production"},
    ],
)
def test_mcp_server_ref(kwargs):
    ref = MCPServerRef("com.example/weather", **kwargs)
    assert ref.name == "com.example/weather"
    assert ref.version == kwargs.get("version")
    assert ref.alias == kwargs.get("alias")
    with pytest.raises(FrozenInstanceError, match="cannot assign"):
        ref.version = "2.0.0"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": None},
        {"name": 123},
        {"name": "https://example.com/mcp"},
        {"name": "invalid"},
        {"name": "com.example/" + "a" * 256},
        {"version": "1.0.0", "alias": "prod"},
        *[
            {field: value}
            for field in ["version", "alias"]
            for value in ["", " ", 1, "a" * 257, "a\x00", "a\ud800"]
        ],
    ],
)
def test_invalid_mcp_server_ref(kwargs):
    with pytest.raises(MlflowException, match="MCP server"):
        MCPServerRef(**{"name": "com.example/weather", **kwargs})
