import pytest

from mlflow.utils.pydantic_utils import IS_PYDANTIC_V2_OR_NEWER

# Skip the entire module if not using pydantic v2
if not IS_PYDANTIC_V2_OR_NEWER:
    pytest.skip(
        "ResponsesAgent and its pydantic classes are not supported in pydantic v1. Skipping test.",
        allow_module_level=True,
    )

from mlflow.types.responses import ResponsesRequest, ResponsesResponse, ResponsesStreamEvent


def test_responses_request_validation():
    with pytest.raises(ValueError, match="content.0.text"):
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

    with pytest.raises(ValueError, match="role"):
        ResponsesRequest(
            **{
                "input": [
                    {
                        "type": "message",
                        "id": "1",
                        "status": "completed",
                        "role": "asdf",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "asdf",
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
    with pytest.raises(ValueError, match="content must not be an empty"):
        ResponsesStreamEvent(
            **{
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                    "id": "1",
                },
            }
        )

    with pytest.raises(ValueError, match="Invalid status"):
        ResponsesStreamEvent(
            **{
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "status": "asdf",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "asdf",
                        }
                    ],
                    "id": "1",
                },
            }
        )

    with pytest.raises(ValueError, match="item.content.0.annotations.0.url"):
        ResponsesStreamEvent(
            **{
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "message",
                    "id": "msg_67ed73ed2c288191b0f0f445e21c66540fbd8030171e9b0c",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "On T",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "start_index": 359,
                                    "end_index": 492,
                                    "title": "NBA roundup:",
                                },
                            ],
                        }
                    ],
                },
            },
        )
    with pytest.raises(ValueError, match="delta"):
        ResponsesStreamEvent(
            **{
                "type": "response.output_text.delta",
                "item_id": "msg_67eda402cba48191a1c35b84af04fc3c0a4363ad71e9395a",
                "output_index": 0,
                "content_index": 0,
            },
        )

    with pytest.raises(ValueError, match="annotation.url"):
        ResponsesStreamEvent(
            **{
                "type": "response.output_text.annotation.added",
                "item_id": "msg_67ed73ed2c288191b0f0f445e21c66540fbd8030171e9b0c",
                "output_index": 1,
                "content_index": 0,
                "annotation_index": 0,
                "annotation": {
                    "type": "url_citation",
                    "start_index": 359,
                    "end_index": 492,
                    "title": "NBA roundup: Wolves overcome Nikola",
                },
            },
        )
