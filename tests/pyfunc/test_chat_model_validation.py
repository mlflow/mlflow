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


def test_dataclass_constructs_nested_types_from_dict():
    response = ChatResponse(**MOCK_RESPONSE)
    assert isinstance(response.usage, TokenUsageStats)
    assert isinstance(response.choices[0], ChatChoice)
    assert isinstance(response.choices[0].message, ChatMessage)


def test_to_dict_converts_nested_dataclasses():
    response = ChatResponse(**MOCK_RESPONSE).to_dict()
    assert isinstance(response["choices"][0], dict)
    assert isinstance(response["usage"], dict)
    assert isinstance(response["choices"][0]["message"], dict)


def test_to_dict_excludes_nones():
    response = ChatResponse(**MOCK_RESPONSE).to_dict()
    assert "name" not in response["choices"][0]["message"]
