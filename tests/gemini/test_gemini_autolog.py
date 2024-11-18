from unittest.mock import patch

import google.generativeai as genai
import pytest

import mlflow
from mlflow.entities.span import SpanType

from tests.tracing.helper import get_traces

CANDIDATES = [{"content": {"parts": [{"text": "test answer"}], "role": "model"}}]

USER_METADATA = {
    "prompt_token_count": 6,
    "candidates_token_count": 6,
    "total_token_count": 6,
    "cached_content_token_count": 0,
}

DUMMY_GENERATE_CONTENT_RESPONSE = {
    "candidates": CANDIDATES,
    "usage_metadata": USER_METADATA,
}

DUMMY_COUNT_TOKENS_RESPONSE = {"total_count": 10}

DUMMY_EMBEDDING_RESPONSE = {"embedding": [1, 2, 3]}


def generate_content(self, contents):
    return DUMMY_GENERATE_CONTENT_RESPONSE


def send_message(self, content):
    return DUMMY_GENERATE_CONTENT_RESPONSE


def count_tokens(self, contents):
    return DUMMY_COUNT_TOKENS_RESPONSE


def embed_content(model, content):
    return DUMMY_EMBEDDING_RESPONSE


def test_generate_content_enable_disable_autolog():
    with patch("google.generativeai.GenerativeModel.generate_content", new=generate_content):
        mlflow.gemini.autolog()
        model = genai.GenerativeModel("gemini-1.5-flash")
        model.generate_content("test content")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 1
        span = traces[0].data.spans[0]
        assert span.name == "GenerativeModel.generate_content"
        assert span.span_type == SpanType.LLM
        assert span.inputs == {"contents": "test content", "model_name": "models/gemini-1.5-flash"}
        assert span.outputs == DUMMY_GENERATE_CONTENT_RESPONSE

        mlflow.gemini.autolog(disable=True)
        model = genai.GenerativeModel("gemini-1.5-flash")
        model.generate_content("test content")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1


def test_generate_content_tracing_with_error():
    with patch(
        "google.generativeai.GenerativeModel.generate_content", side_effect=Exception("dummy error")
    ):
        mlflow.gemini.autolog()
        model = genai.GenerativeModel("gemini-1.5-flash")

        with pytest.raises(Exception, match="dummy error"):
            model.generate_content("test content")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "ERROR"
        assert traces[0].data.spans[0].status.status_code == "ERROR"
        assert traces[0].data.spans[0].status.description == "Exception: dummy error"

        # for preventing potential side effects for other tests
        mlflow.gemini.autolog(disable=True)


def test_chat_session_autolog():
    with patch("google.generativeai.ChatSession.send_message", new=send_message):
        mlflow.gemini.autolog()
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat(history=[])
        chat.send_message("test content")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 1
        span = traces[0].data.spans[0]
        assert span.name == "ChatSession.send_message"
        assert span.span_type == SpanType.CHAT_MODEL
        assert span.inputs == {"content": "test content"}
        assert span.outputs == DUMMY_GENERATE_CONTENT_RESPONSE

        mlflow.gemini.autolog(disable=True)
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat(history=[])
        chat.send_message("test content")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1


def test_count_tokens_autolog():
    with patch("google.generativeai.GenerativeModel.count_tokens", new=count_tokens):
        mlflow.gemini.autolog()
        model = genai.GenerativeModel("gemini-1.5-flash")
        model.count_tokens("test content")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 1
        span = traces[0].data.spans[0]
        assert span.name == "GenerativeModel.count_tokens"
        assert span.span_type == SpanType.LLM
        assert span.inputs == {"contents": "test content", "model_name": "models/gemini-1.5-flash"}
        assert span.outputs == DUMMY_COUNT_TOKENS_RESPONSE

        mlflow.gemini.autolog(disable=True)
        model = genai.GenerativeModel("gemini-1.5-flash")
        model.count_tokens("test content")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1


def test_embed_content_autolog():
    with patch("google.generativeai.embed_content", new=embed_content):
        mlflow.gemini.autolog()
        genai.embed_content(model="models/text-embedding-004", content="Hello World")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 1
        span = traces[0].data.spans[0]
        assert span.name == "embed_content"
        assert span.span_type == SpanType.EMBEDDING
        assert span.inputs == {"content": "Hello World", "model": "models/text-embedding-004"}
        assert span.outputs == DUMMY_EMBEDDING_RESPONSE

        mlflow.gemini.autolog(disable=True)
        genai.embed_content(model="models/text-embedding-004", content="Hello World")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1
