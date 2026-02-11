"""
Temporary test file to verify parallel tool calls work correctly with the fix.
Uses the Databricks workspace client with OpenAI-compatible API.
"""

import json

from databricks.sdk import WorkspaceClient

# Initialize workspace client with default profile (dogfood)
w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

# Define multiple tools that the model can call in parallel
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search for documents matching a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# Message designed to trigger multiple parallel tool calls
messages = [
    {
        "role": "user",
        "content": "I need you to do three things at once: "
        "1) Search for documents about 'machine learning best practices', "
        "2) Get the weather in Seattle, WA, "
        "3) Calculate 42 * 17. "
        "Please call all three tools in parallel.",
    }
]


def test_non_streaming():
    """Test non-streaming response with parallel tool calls."""
    print("=" * 60)
    print("Testing NON-STREAMING parallel tool calls")
    print("=" * 60)

    response = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    print(f"Finish reason: {response.choices[0].finish_reason}")
    print(f"Number of tool calls: {len(response.choices[0].message.tool_calls or [])}")

    if response.choices[0].message.tool_calls:
        for i, tc in enumerate(response.choices[0].message.tool_calls):
            print(f"\nTool call {i + 1}:")
            print(f"  ID: {tc.id}")
            print(f"  Function: {tc.function.name}")
            print(f"  Arguments: {tc.function.arguments}")
            # Verify JSON is valid
            try:
                parsed = json.loads(tc.function.arguments)
                print(f"  Parsed OK: {parsed}")
            except json.JSONDecodeError as e:
                print(f"  JSON ERROR: {e}")


def test_streaming():
    """Test streaming response with parallel tool calls."""
    print("\n" + "=" * 60)
    print("Testing STREAMING parallel tool calls")
    print("=" * 60)

    stream = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=True,
    )

    # Collect tool calls by index (this is what the fix does)
    tool_calls: dict[int, dict] = {}

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc_delta.id,
                        "function": {
                            "name": getattr(tc_delta.function, "name", "") or "",
                            "arguments": getattr(tc_delta.function, "arguments", "") or "",
                        },
                    }
                else:
                    if tc_delta.function and tc_delta.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments
                    if tc_delta.id:
                        tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        tool_calls[idx]["function"]["name"] = tc_delta.function.name

    print(f"Number of tool calls collected: {len(tool_calls)}")

    for idx in sorted(tool_calls.keys()):
        tc = tool_calls[idx]
        print(f"\nTool call {idx + 1}:")
        print(f"  ID: {tc['id']}")
        print(f"  Function: {tc['function']['name']}")
        print(f"  Arguments: {tc['function']['arguments']}")
        # Verify JSON is valid
        try:
            parsed = json.loads(tc["function"]["arguments"])
            print(f"  Parsed OK: {parsed}")
        except json.JSONDecodeError as e:
            print(f"  JSON ERROR: {e}")


def test_streaming_with_mlflow_converter():
    """Test streaming with MLflow's output_to_responses_items_stream converter."""
    print("\n" + "=" * 60)
    print("Testing STREAMING with MLflow converter")
    print("=" * 60)

    from mlflow.types.responses import output_to_responses_items_stream

    stream = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=True,
    )

    # Collect raw chunks first
    raw_chunks = []
    for chunk in stream:
        chunk_dict = chunk.model_dump()
        raw_chunks.append(chunk_dict)

    # Print raw chunks for test case
    print("\n--- RAW CHUNKS (for unit test) ---")
    print("chunks = [")
    for chunk in raw_chunks:
        # Filter to only relevant fields for the test
        filtered = {
            "id": chunk.get("id"),
            "choices": chunk.get("choices"),
            "object": chunk.get("object"),
        }
        print(f"    {filtered},")
    print("]")

    # Now convert using MLflow converter
    aggregator = []
    events = list(output_to_responses_items_stream(iter(raw_chunks), aggregator))

    print("\n--- CONVERTED EVENTS (for unit test) ---")
    print("expected_output = [")
    for event in events:
        print(f"    {event},")
    print("]")

    print(f"\nNumber of events: {len(events)}")
    print(f"Number of aggregated items: {len(aggregator)}")

    for i, item in enumerate(aggregator):
        if item.get("type") == "function_call":
            print(f"\nFunction call {i + 1}:")
            print(f"  Call ID: {item.get('call_id')}")
            print(f"  Name: {item.get('name')}")
            print(f"  Arguments: {item.get('arguments')}")
            # Verify JSON is valid
            try:
                parsed = json.loads(item.get("arguments", "{}"))
                print(f"  Parsed OK: {parsed}")
            except json.JSONDecodeError as e:
                print(f"  JSON ERROR: {e}")


if __name__ == "__main__":
    test_non_streaming()
    test_streaming()
    test_streaming_with_mlflow_converter()
