from unittest.mock import patch

import google.generativeai as genai

import mlflow
import pytest
from mlflow.entities.span import SpanType
from tests.tracing.helper import get_traces

CANDIDATES = [{
                "content": {
                "parts": [
                    {
                    "text": "test answer"
                    }
                ],
                "role": "model"
                }
            }]

USER_METADATA = {  'prompt_token_count': 6,
                    'candidates_token_count': 6,
                    'total_token_count': 6,
                    'cached_content_token_count': 0
                }

def test_enable_disable_autolog():
    genai.GenerativeModel.generate_content = lambda self,contents: {
                   "candidates": CANDIDATES,
                   'usage_metadata': USER_METADATA,
                }
    mlflow.gemini.autolog()
    model = genai.GenerativeModel('gemini-1.5-flash')
    model.generate_content("test content")

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].info.execution_time_ms > 0
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "GenerativeModel"
    assert span.span_type == SpanType.LLM
    assert span.inputs["contents"] == "test content"
    assert span.outputs["candidates"] == CANDIDATES
    assert span.outputs["usage_metadata"] == USER_METADATA

    mlflow.gemini.autolog(disable=True)
    model = genai.GenerativeModel('gemini-1.5-flash')
    model.generate_content("test content")

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1


def test_tracing_with_error():
    with patch("google.generativeai.GenerativeModel.generate_content", side_effect=Exception("dummy error")):
        mlflow.gemini.autolog()
        model = genai.GenerativeModel('gemini-1.5-flash')

        with pytest.raises(Exception, match="dummy error"):
            model.generate_content("test content")

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    assert traces[0].info.execution_time_ms > 0
    assert traces[0].data.spans[0].status.status_code == "ERROR"
    assert traces[0].data.spans[0].status.description == "Exception: dummy error"
