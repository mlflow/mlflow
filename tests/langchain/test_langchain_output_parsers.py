from mlflow.langchain.output_parsers import ChatCompletionsOutputParser, StringResponseOutputParser


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
