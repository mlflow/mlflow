import pytest

from mlflow.genai.judges.utils.formatting_utils import format_available_tools, format_tools_called
from mlflow.genai.utils.type import FunctionCall
from mlflow.types.chat import (
    ChatTool,
    FunctionParams,
    FunctionToolDefinition,
    ParamProperty,
)


@pytest.mark.parametrize(
    ("tools", "expected"),
    [
        pytest.param(
            [
                ChatTool(
                    type="function",
                    function=FunctionToolDefinition(
                        name="get_weather",
                        description="Get current weather for a location",
                    ),
                )
            ],
            "- get_weather: Get current weather for a location",
            id="basic",
        ),
        pytest.param(
            [
                ChatTool(
                    type="function",
                    function=FunctionToolDefinition(
                        name="search",
                        description="Search for information",
                        parameters=FunctionParams(
                            properties={
                                "query": ParamProperty(
                                    type="string", description="The search query"
                                ),
                                "max_results": ParamProperty(
                                    type="integer", description="Maximum number of results"
                                ),
                            },
                            required=["query"],
                        ),
                    ),
                )
            ],
            (
                "- search: Search for information\n"
                "    - query (required): string - The search query\n"
                "    - max_results (optional): integer - Maximum number of results"
            ),
            id="with_parameters",
        ),
        pytest.param(
            [
                ChatTool(
                    type="function",
                    function=FunctionToolDefinition(name="tool1", description="First tool"),
                ),
                ChatTool(
                    type="function",
                    function=FunctionToolDefinition(name="tool2", description="Second tool"),
                ),
            ],
            "- tool1: First tool\n\n- tool2: Second tool",
            id="multiple",
        ),
        pytest.param(
            [],
            "No tools available",
            id="empty",
        ),
        pytest.param(
            [
                ChatTool(type="function", function=None),
                ChatTool(
                    type="function",
                    function=FunctionToolDefinition(name="valid_tool", description="Valid tool"),
                ),
            ],
            "- valid_tool: Valid tool",
            id="missing_function",
        ),
        pytest.param(
            [
                ChatTool(
                    type="function",
                    function=FunctionToolDefinition(
                        name="calc",
                        parameters=FunctionParams(
                            properties={
                                "x": ParamProperty(type="number"),
                                "y": ParamProperty(type="number"),
                            },
                            required=["x", "y"],
                        ),
                    ),
                )
            ],
            "- calc\n    - x (required): number\n    - y (required): number",
            id="parameter_without_description",
        ),
    ],
)
def test_format_available_tools(tools, expected):
    result = format_available_tools(tools)
    assert result == expected


@pytest.mark.parametrize(
    ("tools_called", "expected"),
    [
        pytest.param(
            [
                FunctionCall(
                    name="get_weather",
                    arguments={"city": "Paris"},
                    outputs="Sunny, 22°C",
                )
            ],
            (
                "Tool Call 1: get_weather\n"
                "  Input Arguments: {'city': 'Paris'}\n"
                "  Output: Sunny, 22°C"
            ),
            id="basic",
        ),
        pytest.param(
            [
                FunctionCall(
                    name="search",
                    arguments={"query": "capital of France"},
                    outputs="Paris",
                ),
                FunctionCall(
                    name="translate",
                    arguments={"text": "Paris", "target": "es"},
                    outputs="París",
                ),
            ],
            (
                "Tool Call 1: search\n"
                "  Input Arguments: {'query': 'capital of France'}\n"
                "  Output: Paris\n"
                "\n"
                "Tool Call 2: translate\n"
                "  Input Arguments: {'text': 'Paris', 'target': 'es'}\n"
                "  Output: París"
            ),
            id="multiple",
        ),
        pytest.param(
            [
                FunctionCall(
                    name="get_weather",
                    arguments={"city": "InvalidCity"},
                    outputs=None,
                    exception="ValueError: City not found",
                )
            ],
            (
                "Tool Call 1: get_weather\n"
                "  Input Arguments: {'city': 'InvalidCity'}\n"
                "  Output: (no output)\n"
                "  Exception: ValueError: City not found"
            ),
            id="with_exception",
        ),
        pytest.param(
            [
                FunctionCall(
                    name="stream_data",
                    arguments={"source": "api"},
                    outputs={"items": [1, 2]},
                    exception="TimeoutError: Connection lost",
                )
            ],
            (
                "Tool Call 1: stream_data\n"
                "  Input Arguments: {'source': 'api'}\n"
                "  Output: {'items': [1, 2]}\n"
                "  Exception: TimeoutError: Connection lost"
            ),
            id="with_partial_output_and_exception",
        ),
        pytest.param(
            [
                FunctionCall(
                    name="send_notification",
                    arguments={"message": "Hello"},
                    outputs=None,
                )
            ],
            (
                "Tool Call 1: send_notification\n"
                "  Input Arguments: {'message': 'Hello'}\n"
                "  Output: (no output)"
            ),
            id="no_output",
        ),
        pytest.param(
            [],
            "No tools called",
            id="empty",
        ),
        pytest.param(
            [
                FunctionCall(
                    name="get_time",
                    arguments=None,
                    outputs="12:00 PM",
                )
            ],
            "Tool Call 1: get_time\n  Input Arguments: {}\n  Output: 12:00 PM",
            id="empty_arguments",
        ),
    ],
)
def test_format_tools_called(tools_called, expected):
    result = format_tools_called(tools_called)
    assert result == expected
