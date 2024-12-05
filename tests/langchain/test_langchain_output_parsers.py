from langchain_core.messages.base import BaseMessage
from langchain_core.runnables.config import RunnableConfig

from mlflow.langchain.output_parsers import (
    ChatCompletionOutputParser,
    ChatCompletionsOutputParser,
    StringResponseOutputParser,
)


def test_chatcompletions_output_parser_parse_response():
    parser = ChatCompletionsOutputParser()
    message = "The weather today is"

    parsed_response = parser.parse(message)
    assert parsed_response == {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "The weather today is", "role": "assistant"},
            }
        ],
        "object": "chat.completion",
    }


def test_chatcompletions_output_parser_is_lc_serializable():
    parser = StringResponseOutputParser()
    message = "The weather today is"

    parsed_response = parser.parse(message)
    assert parsed_response == {"content": "The weather today is"}


def test_chatcompletion_output_parser_parse_response():
    parser = ChatCompletionOutputParser()
    message = "The weather today is"

    parsed_response = parser.parse(message)
    assert isinstance(parsed_response["created"], int)
    del parsed_response["created"]

    assert parsed_response == {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The weather today is",
                    "role": "assistant",
                },
            }
        ],
        "object": "chat.completion",
    }

    streaming_messages = ["The ", "weather ", "today ", "is"]
    base_messages = [BaseMessage(content=m, type="test") for m in streaming_messages]
    streaming_chunks = parser.transform(base_messages, RunnableConfig())
    for i, chunk in enumerate(streaming_chunks):
        assert isinstance(chunk["created"], int)
        del chunk["created"]
        assert chunk == {
            "choices": [
                {
                    "delta": {
                        "content": streaming_messages[i],
                        "role": "assistant",
                    },
                    "index": 0,
                }
            ],
            "object": "chat.completion.chunk",
        }
