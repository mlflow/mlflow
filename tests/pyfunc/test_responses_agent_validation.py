import pytest

from mlflow.types.responses import ResponsesResponse, ResponsesStreamEvent


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
