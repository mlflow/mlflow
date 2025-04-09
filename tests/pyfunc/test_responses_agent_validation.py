import pytest

from mlflow.types.responses import ResponsesRequest, ResponsesResponse, ResponsesStreamEvent


def test_responses_request_validation():
    with pytest.raises(ValueError, match="input.0.content.0.text"):
        ResponsesRequest(
            **{
                "input": [
                    {
                        "type": "message",
                        "id": "1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                            }
                        ],
                    }
                ],
            }
        )


def test_responses_response_validation():
    with pytest.raises(ValueError, match="output.0.content.0.text"):
        ResponsesResponse(
            **{
                "output": [
                    {
                        "type": "message",
                        "id": "1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                            }
                        ],
                    }
                ],
            }
        )


def test_responses_stream_event_validation():
    with pytest.raises(ValueError, match="output.0.content.0.text"):
        ResponsesStreamEvent(
            **{
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            }
        )
