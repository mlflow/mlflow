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
                        "bytes": [33],
                        "top_logprobs": [
                            {"token": "!", "logprob": -0.02380986, "bytes": [33]},
                            {
                                "token": " there",
                                "logprob": -3.787621,
                                "bytes": [32, 116, 104, 101, 114, 101],
                            },
                        ],
                    },
                    {
                        "token": " How",
                        "logprob": -0.000054669687,
                        "bytes": [32, 72, 111, 119],
                        "top_logprobs": [
                            {
                                "token": " How",
                                "logprob": -0.000054669687,
                                "bytes": [32, 72, 111, 119],
                            },
                            {"token": "<|end|>", "logprob": -10.953937, "bytes": None},
                        ],
                    },
                    {
                        "token": " can",
                        "logprob": -0.015801601,
                        "bytes": [32, 99, 97, 110],
                        "top_logprobs": [
                            {"token": " can", "logprob": -0.015801601, "bytes": [32, 99, 97, 110]},
                            {"token": " may", "logprob": -4.161023, "bytes": [32, 109, 97, 121]},
                        ],
                    },
                    {
                        "token": " I",
                        "logprob": -3.7697225e-6,
                        "bytes": [32, 73],
                        "top_logprobs": [
                            {"token": " I", "logprob": -3.7697225e-6, "bytes": [32, 73]},
                            {
                                "token": " assist",
                                "logprob": -13.596657,
                                "bytes": [32, 97, 115, 115, 105, 115, 116],
                            },
                        ],
                    },
                    {
                        "token": " assist",
                        "logprob": -0.04571125,
                        "bytes": [32, 97, 115, 115, 105, 115, 116],
                        "top_logprobs": [
                            {
                                "token": " assist",
                                "logprob": -0.04571125,
                                "bytes": [32, 97, 115, 115, 105, 115, 116],
                            },
                            {
                                "token": " help",
                                "logprob": -3.1089056,
                                "bytes": [32, 104, 101, 108, 112],
                            },
                        ],
                    },
                    {
                        "token": " you",
                        "logprob": -5.4385737e-6,
                        "bytes": [32, 121, 111, 117],
                        "top_logprobs": [
                            {
                                "token": " you",
                                "logprob": -5.4385737e-6,
                                "bytes": [32, 121, 111, 117],
                            },
                            {
                                "token": " today",
                                "logprob": -12.807695,
                                "bytes": [32, 116, 111, 100, 97, 121],
                            },
                        ],
                    },
                    {
                        "token": " today",
                        "logprob": -0.0040071653,
                        "bytes": [32, 116, 111, 100, 97, 121],
                        "top_logprobs": [
                            {
                                "token": " today",
                                "logprob": -0.0040071653,
                                "bytes": [32, 116, 111, 100, 97, 121],
                            },
                            {"token": "?", "logprob": -5.5247097, "bytes": [63]},
                        ],
                    },
                    {
                        "token": "?",
                        "logprob": -0.0008108172,
                        "bytes": [63],
                        "top_logprobs": [
                            {"token": "?", "logprob": -0.0008108172, "bytes": [63]},
                            {"token": "?\n", "logprob": -7.184561, "bytes": [63, 10]},
                        ],
                    },
                ]
            },
            "finish_reason": "stop",
        }
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
