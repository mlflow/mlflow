import pytest

from mlflow.types.llm import ChatChoice, ChatMessage, ChatRequest, ChatResponse, TokenUsageStats

MOCK_RESPONSE = {
    "id": "123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "MyChatModel",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "hello",
            },
            "finish_reason": "stop",
        },
        {
            "index": 1,
            "message": {
                "role": "user",
                "content": "world",
            },
            "finish_reason": "stop",
        },
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 10,
        "total_tokens": 20,
    },
}

MOCK_OPENAI_CHAT_COMPLETION_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1702685778,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I assist you today?"},
            "logprobs": {
                "content": [
                    {
                        "token": "Hello",
                        "logprob": -0.31725305,
                        "bytes": [72, 101, 108, 108, 111],
                        "top_logprobs": [
                            {
                                "token": "Hello",
                                "logprob": -0.31725305,
                                "bytes": [72, 101, 108, 108, 111],
                            },
                            {"token": "Hi", "logprob": -1.3190403, "bytes": [72, 105]},
                        ],
                    },
                    {
                        "token": "!",
                        "logprob": -0.02380986,
                        "bytes": None,
                        "top_logprobs": [
                            {"token": "!", "logprob": -0.02380986, "bytes": [33]},
                            {
                                "token": " there",
                                "logprob": -3.787621,
                                "bytes": None,
                            },
                        ],
                    },
                ]
            },
            "finish_reason": "stop",
        },
        {
            "index": 1,
            "message": {"role": "user", "content": "I need help with my computer."},
            "logprobs": None,
            "finish_reason": "stop",
        },
        {
            "index": 2,
            "message": {"role": "assistant", "content": "Sure! What seems to be the problem?"},
            "logprobs": {
                "content": None,
            },
            "finish_reason": "stop",
        },
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 9, "total_tokens": 18},
}


MOCK_OPENAI_CHAT_REFUSAL_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1721596428,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "refusal": "I'm sorry, I cannot assist with that request.",
            },
            "logprobs": None,
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 81, "completion_tokens": 11, "total_tokens": 92},
}


@pytest.mark.parametrize(
    ("data", "error", "match"),
    [
        ({"content": "hello"}, TypeError, "required positional argument"),  # missing required field
        (
            {"role": "user", "content": "hello", "name": 1},
            ValueError,
            "`name` must be of type str",
        ),  # field of wrong type
        (
            {"role": "user", "refusal": "I can't answer that.", "content": "hi"},
            ValueError,
            "Both `content` and `refusal` cannot be set",
        ),  # conflicting schema
        (
            {"role": "user", "name": "name"},
            ValueError,
            "`content` is required",
        ),  # missing one-of required field
    ],
)
def test_chat_message_throws_on_invalid_data(data, error, match):
    with pytest.raises(error, match=match):
        ChatMessage.from_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "hello", "name": "world"},
    ],
)
def test_chat_message_succeeds_on_valid_data(data):
    assert ChatMessage.from_dict(data).to_dict() == data


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"messages": "not a list"}, "`messages` must be a list"),
        (
            {"messages": ["not a dict"]},
            "Items in `messages` must all have the same type: ChatMessage or dict",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "not all the same"},
                    ChatMessage.from_dict({"role": "user", "content": "hello"}),
                ]
            },
            "Items in `messages` must all have the same type: ChatMessage or dict",
        ),
    ],
)
def test_list_validation_throws_on_invalid_lists(data, match):
    with pytest.raises(ValueError, match=match):
        ChatRequest.from_dict(data)


@pytest.mark.parametrize(
    "sample_output",
    [MOCK_RESPONSE, MOCK_OPENAI_CHAT_COMPLETION_RESPONSE, MOCK_OPENAI_CHAT_REFUSAL_RESPONSE],
)
def test_dataclass_constructs_nested_types_from_dict(sample_output):
    response = ChatResponse.from_dict(sample_output)
    assert isinstance(response.usage, TokenUsageStats)
    assert isinstance(response.choices[0], ChatChoice)
    assert isinstance(response.choices[0].message, ChatMessage)


@pytest.mark.parametrize(
    "sample_output",
    [MOCK_RESPONSE, MOCK_OPENAI_CHAT_COMPLETION_RESPONSE, MOCK_OPENAI_CHAT_REFUSAL_RESPONSE],
)
def test_to_dict_converts_nested_dataclasses(sample_output):
    response = ChatResponse.from_dict(sample_output).to_dict()
    assert isinstance(response["choices"][0], dict)
    assert isinstance(response["usage"], dict)
    assert isinstance(response["choices"][0]["message"], dict)


def test_to_dict_excludes_nones():
    response = ChatResponse.from_dict(MOCK_RESPONSE).to_dict()
    assert "name" not in response["choices"][0]["message"]


def test_chat_response_defaults():
    tokens = TokenUsageStats()
    message = ChatMessage("user", "Hello")
    choice = ChatChoice(0, message)
    response = ChatResponse([choice], tokens)

    assert response.usage.prompt_tokens is None
    assert response.usage.completion_tokens is None
    assert response.usage.total_tokens is None
    assert response.model is None
    assert response.id is None
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        (1, r"Expected `metadata` to be a dictionary, received `int`"),
        ({"nested": {"dict": "input"}}, r"received value of type `dict` in `metadata\['nested'\]`"),
        ({1: "example"}, r"received key of type `int` \(key: 1\)"),
    ],
)
def test_chat_request_metadata_must_be_string_map(metadata, match):
    message = ChatMessage("user", "Hello")
    with pytest.raises(ValueError, match=match):
        ChatRequest(messages=[message], metadata=metadata)


@pytest.mark.parametrize(
    ("cls", "data", "match"),
    [
        (
            ChatChoice,
            {"index": 0, "message": 123},
            "Expected `message` to be either an instance of `ChatMessage` or a dict",
        ),
        (
            ChatResponse,
            {"choices": [], "usage": 123},
            "Expected `usage` to be either an instance of `TokenUsageStats` or a dict",
        ),
    ],
)
def test_convert_dataclass_throws_on_invalid_data(cls, data, match):
    with pytest.raises(ValueError, match=match):
        cls.from_dict(data)


@pytest.mark.parametrize(
    ("cls", "data"),
    [
        (ChatMessage, {"role": "user", "content": "hello", "extra": "field"}),
        (
            TokenUsageStats,
            {
                "completion_tokens": 10,
                "prompt_tokens": 57,
                "total_tokens": 67,
                # this field is not in the TokenUsageStats schema
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        ),
    ],
)
def test_from_dict_ingores_extra_fields(cls, data):
    assert isinstance(cls.from_dict(data), cls)
