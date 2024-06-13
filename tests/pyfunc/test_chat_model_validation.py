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
    "model": "gpt-3.5-turbo-0125",
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


@pytest.mark.parametrize(
    ("data", "error", "match"),
    [
        ({"role": "user"}, TypeError, "required positional argument"),  # missing required field
        (
            {"role": "user", "content": "hello", "name": 1},
            ValueError,
            "`name` must be of type str",
        ),  # field of wrong type
        (
            {"role": "user", "content": "hello", "extra": "field"},
            TypeError,
            "unexpected keyword",
        ),  # extra field
    ],
)
def test_chat_message_throws_on_invalid_data(data, error, match):
    with pytest.raises(error, match=match):
        ChatMessage(**data)


@pytest.mark.parametrize(
    "data",
    [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "hello", "name": "world"},
    ],
)
def test_chat_message_succeeds_on_valid_data(data):
    assert ChatMessage(**data).to_dict() == data


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"messages": "not a list"}, "`messages` must be a list"),
        (
            {"messages": ["not a dict"]},
            "Items in `messages` must all have the same type: ChatMessage or dict",
        ),
        ({"messages": [{"bad": "key"}]}, "unexpected keyword argument 'bad'"),
        (
            {
                "messages": [
                    {"role": "user", "content": "not all the same"},
                    ChatMessage(**{"role": "user", "content": "hello"}),
                ]
            },
            "Items in `messages` must all have the same type: ChatMessage or dict",
        ),
    ],
)
def test_list_validation_throws_on_invalid_lists(data, match):
    with pytest.raises(ValueError, match=match):
        ChatRequest(**data)


@pytest.mark.parametrize("sample_output", [MOCK_RESPONSE, MOCK_OPENAI_CHAT_COMPLETION_RESPONSE])
def test_dataclass_constructs_nested_types_from_dict(sample_output):
    response = ChatResponse(**sample_output)
    assert isinstance(response.usage, TokenUsageStats)
    assert isinstance(response.choices[0], ChatChoice)
    assert isinstance(response.choices[0].message, ChatMessage)


@pytest.mark.parametrize("sample_output", [MOCK_RESPONSE, MOCK_OPENAI_CHAT_COMPLETION_RESPONSE])
def test_to_dict_converts_nested_dataclasses(sample_output):
    response = ChatResponse(**sample_output).to_dict()
    assert isinstance(response["choices"][0], dict)
    assert isinstance(response["usage"], dict)
    assert isinstance(response["choices"][0]["message"], dict)


def test_to_dict_excludes_nones():
    response = ChatResponse(**MOCK_RESPONSE).to_dict()
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
