from mlflow.gateway.providers.gemini import GeminiAdapter


def test_chat_to_model_does_not_reuse_thought_signature_for_duplicate_id():
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    for signature in ("sig_1", None):
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
        if signature is not None:
            tool_call["thought_signature"] = signature
        messages.extend([
            {"role": "assistant", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temperature": 20}'},
        ])

    result = GeminiAdapter.chat_to_model({"messages": messages}, config=None)

    model_parts = [
        part for turn in result["contents"] if turn["role"] == "model" for part in turn["parts"]
    ]
    assert [part.get("thoughtSignature") for part in model_parts] == ["sig_1", None]
    assert "thoughtSignature" not in model_parts[1]
    assert [
        part["functionResponse"]["name"]
        for turn in result["contents"]
        for part in turn["parts"]
        if "functionResponse" in part
    ] == ["get_weather", "get_weather"]
