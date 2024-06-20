# TODO: ADD MORE TESTS
import json
from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.providers.openai import OpenAIProvider
from mlflow.gateway.schemas import chat
from mlflow.gateway.uc_function_utils import uc_type_to_json_schema_type

from tests.gateway.tools import (
    MockAsyncResponse,
    MockHttpClient,
)


def chat_config():
    return {
        "name": "chat",
        "route_type": "llm/v1/chat",
        "model": {
            "provider": "openai",
            "name": "gpt-3.5-turbo",
            "config": {
                "openai_api_base": "https://api.openai.com/v1",
                "openai_api_key": "key",
            },
        },
    }


def test_uc_type_to_json_schema_type():
    # TODO: Improve the test coverage

    # Test with primitive types
    assert uc_type_to_json_schema_type("long") == {"type": "integer"}
    assert uc_type_to_json_schema_type("boolean") == {"type": "boolean"}
    assert uc_type_to_json_schema_type("date") == {"type": "string", "format": "date"}

    # Test with complex types
    assert uc_type_to_json_schema_type({"type": "array", "elementType": "integer"}) == {
        "type": "array",
        "items": {"type": "integer"},
    }
    assert uc_type_to_json_schema_type(
        {"type": "map", "keyType": "string", "valueType": "integer"}
    ) == {"type": "object", "additionalProperties": {"type": "integer"}}
    assert uc_type_to_json_schema_type(
        {"type": "struct", "fields": [{"name": "field1", "type": "integer"}]}
    ) == {"type": "object", "properties": {"field1": {"type": "integer"}}}

    # Test with unsupported types
    with pytest.raises(TypeError, match=r"Type interval is not supported\."):
        uc_type_to_json_schema_type("interval")

    with pytest.raises(TypeError, match=r"Unknown type unknown. Try upgrading this package\."):
        uc_type_to_json_schema_type("unknown")

    with pytest.raises(TypeError, match=r"Only support STRING key type for MAP but got integer\."):
        uc_type_to_json_schema_type({"type": "map", "keyType": "integer", "valueType": "integer"})


@pytest.mark.asyncio
async def test_uc_functions(monkeypatch):
    from databricks.sdk.service.catalog import ColumnTypeName
    from databricks.sdk.service.sql import StatementState

    monkeypatch.setenv("MLFLOW_ENABLE_UC_FUNCTIONS", "true")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "1234")

    config = chat_config()
    first_resp = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0301",
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_id",
                            "function": {
                                "arguments": '{"x": 1, "y": 2}',
                                "name": "test__uc__func",
                            },
                            "type": "function",
                        },
                    ],
                },
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }

    second_resp = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0301",
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "1 + 2 = 3",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }

    mock_client_session = MockHttpClient()
    mock_client_session.post.side_effect = [
        MockAsyncResponse(first_resp),
        MockAsyncResponse(second_resp),
    ]

    params = []
    for name, typ in [("x", "integer"), ("y", "integer")]:
        p = mock.MagicMock()
        p.type_json = json.dumps({"type": typ})
        p.name = name
        params.append(p)

    mock_func_info = mock.MagicMock()
    mock_func_info.input_params.parameters = params
    mock_func_info.data_type = ColumnTypeName.INT
    mock_func_info.name = "func"
    mock_func_info.catalog_name = "test"
    mock_func_info.schema_name = "uc"
    mock_func_info.full_name = "test.uc.func"

    mock_workspace_client = mock.MagicMock()
    mock_workspace_client.functions.get.return_value = mock_func_info

    mock_statement_result = mock.MagicMock()
    mock_statement_result.result.data_array = [[3]]
    mock_statement_result.status.state = StatementState.SUCCEEDED
    mock_statement_result.manifest.truncated = "manifest"
    mock_workspace_client.statement_execution.execute_statement.return_value = mock_statement_result

    with mock.patch("aiohttp.ClientSession", return_value=mock_client_session), mock.patch(
        "mlflow.gateway.providers.openai._get_workspace_client", return_value=mock_workspace_client
    ) as mock_workspace_client:
        provider = OpenAIProvider(RouteConfig(**config))
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a joke",
                }
            ],
            "temperature": 0.5,
            "tools": [
                {
                    "type": "uc_function",
                    "uc_function": {
                        "name": "test.uc.func",
                    },
                }
            ],
        }
        response = await provider.chat(chat.RequestPayload(**payload))
        assert mock_client_session.post.call_count == 2

        assert jsonable_encoder(response) == {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-3.5-turbo-0301",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": r"""
<uc_function_call>
{
  "id": "call_id",
  "name": "test.uc.func",
  "arguments": "{\"x\": 1, \"y\": 2}"
}
</uc_function_call>

<uc_function_result>
{
  "tool_call_id": "call_id",
  "content": "{\"format\": \"SCALAR\", \"value\": \"3\", \"truncated\": \"manifest\"}"
}
</uc_function_result>

1 + 2 = 3""".lstrip(),
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            # The server makes two requests, the token usage should be doubled
            "usage": {"prompt_tokens": 13 * 2, "completion_tokens": 7 * 2, "total_tokens": 20 * 2},
        }


@pytest.mark.asyncio
async def test_uc_functions_user_defined_functions(monkeypatch):
    from databricks.sdk.service.catalog import ColumnTypeName
    from databricks.sdk.service.sql import StatementState

    monkeypatch.setenv("MLFLOW_ENABLE_UC_FUNCTIONS", "true")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "1234")

    config = chat_config()
    first_resp = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0301",
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_id_1",
                            "function": {
                                "arguments": '{"x": 1, "y": 2}',
                                "name": "test__uc__func",
                            },
                            "type": "function",
                        },
                        {
                            "id": "call_id_2",
                            "function": {
                                "arguments": '{"x": 3, "y": 4}',
                                "name": "multiply",
                            },
                            "type": "function",
                        },
                    ],
                },
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }

    second_resp = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0301",
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "1 + 2 = 3",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }

    mock_client_session = MockHttpClient()
    mock_client_session.post.side_effect = [
        MockAsyncResponse(first_resp),
        MockAsyncResponse(second_resp),
    ]

    params = []
    for name, typ in [("x", "integer"), ("y", "integer")]:
        p = mock.MagicMock()
        p.type_json = json.dumps({"type": typ})
        p.name = name
        params.append(p)

    mock_func_info = mock.MagicMock()
    mock_func_info.input_params.parameters = params
    mock_func_info.data_type = ColumnTypeName.INT
    mock_func_info.name = "func"
    mock_func_info.catalog_name = "test"
    mock_func_info.schema_name = "uc"
    mock_func_info.full_name = "test.uc.func"

    mock_workspace_client = mock.MagicMock()
    mock_workspace_client.functions.get.return_value = mock_func_info

    mock_statement_result = mock.MagicMock()
    mock_statement_result.result.data_array = [[3]]
    mock_statement_result.status.state = StatementState.SUCCEEDED
    mock_statement_result.manifest.truncated = "manifest"
    mock_workspace_client.statement_execution.execute_statement.return_value = mock_statement_result

    with mock.patch("aiohttp.ClientSession", return_value=mock_client_session), mock.patch(
        "mlflow.gateway.providers.openai._get_workspace_client", return_value=mock_workspace_client
    ) as mock_workspace_client:
        provider = OpenAIProvider(RouteConfig(**config))
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1 + 2? What is 3 * 4?",
                }
            ],
            "temperature": 0.5,
            "tools": [
                {
                    "type": "uc_function",
                    "uc_function": {
                        "name": "test.uc.func",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "description": "Multiply numbers",
                        "name": "multiply",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "x": {
                                    "type": "integer",
                                    "description": "First number",
                                },
                                "y": {
                                    "type": "integer",
                                    "description": "Second number",
                                },
                            },
                            "required": ["x", "y"],
                        },
                    },
                },
            ],
        }
        response = await provider.chat(chat.RequestPayload(**payload))
        assert mock_client_session.post.call_count == 1

        assert jsonable_encoder(response) == {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-3.5-turbo-0301",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": r"""
<uc_function_call>
{
  "id": "call_id_1",
  "name": "test.uc.func",
  "arguments": "{\"x\": 1, \"y\": 2}"
}
</uc_function_call>

<uc_function_result>
{
  "tool_call_id": "call_id_1",
  "content": "{\"format\": \"SCALAR\", \"value\": \"3\", \"truncated\": \"manifest\"}"
}
</uc_function_result>""".lstrip(),
                        "tool_calls": [
                            {
                                "id": "call_id_2",
                                "type": "function",
                                "function": {
                                    "arguments": '{"x": 3, "y": 4}',
                                    "name": "multiply",
                                },
                            },
                        ],
                    },
                    "finish_reason": None,
                }
            ],
            "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        }
