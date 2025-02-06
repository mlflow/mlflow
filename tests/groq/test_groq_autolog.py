import os
from unittest.mock import patch

import groq
from groq.types.audio.transcription import Transcription
from groq.types.audio.translation import Translation
from groq.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)

import mlflow.groq
from mlflow.entities.span import SpanType

from tests.tracing.helper import get_traces

DUMMY_CHAT_COMPLETION_REQUEST = {
    "model": "test_model",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "test message"}],
}

DUMMY_CHAT_COMPLETION_RESPONSE = ChatCompletion(
    id="chatcmpl-test-id",
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content="test response", role="assistant", function_call=None, tool_calls=None
            ),
        )
    ],
    created=1733574047,
    model="llama3-8b-8192",
    object="chat.completion",
    system_fingerprint="fp_test",
    usage=CompletionUsage(
        completion_tokens=648,
        prompt_tokens=20,
        total_tokens=668,
        completion_time=0.54,
        prompt_time=0.000181289,
        queue_time=0.012770949,
        total_time=0.540181289,
    ),
    x_groq={"id": "req_test"},
)


@patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
@patch("groq._client.Groq.post", return_value=DUMMY_CHAT_COMPLETION_RESPONSE)
def test_chat_completion_autolog(mock_post):
    mlflow.groq.autolog()
    client = groq.Groq()
    client.chat.completions.create(**DUMMY_CHAT_COMPLETION_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Completions"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_CHAT_COMPLETION_REQUEST
    assert span.outputs == DUMMY_CHAT_COMPLETION_RESPONSE.to_dict()

    mlflow.groq.autolog(disable=True)
    client = groq.Groq()
    client.chat.completions.create(**DUMMY_CHAT_COMPLETION_REQUEST)

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1


BINARY_CONTENT = b"\x00\x00\x00\x14ftypM4A \x00\x00\x00\x00mdat\x00\x01\x02\x03"

DUMMY_AUDIO_TRANSCRIPTION_REQUEST = {
    "file": ("test_audio.m4a", BINARY_CONTENT),
    "model": "whisper-large-v3-turbo",
}

DUMMY_AUDIO_TRANSCRIPTION_RESPONSE = Transcription(text="Test audio", x_groq={"id": "req_test"})


@patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
@patch("groq._client.Groq.post", return_value=DUMMY_AUDIO_TRANSCRIPTION_RESPONSE)
def test_audio_transcription_autolog(mock_post):
    mlflow.groq.autolog()
    client = groq.Groq()
    client.audio.transcriptions.create(**DUMMY_AUDIO_TRANSCRIPTION_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Transcriptions"
    assert span.span_type == SpanType.LLM
    assert span.inputs["file"][0] == DUMMY_AUDIO_TRANSCRIPTION_REQUEST["file"][0]
    assert span.inputs["file"][1] == str(DUMMY_AUDIO_TRANSCRIPTION_REQUEST["file"][1])
    assert span.inputs["model"] == DUMMY_AUDIO_TRANSCRIPTION_REQUEST["model"]
    assert span.outputs == DUMMY_AUDIO_TRANSCRIPTION_RESPONSE.to_dict()

    mlflow.groq.autolog(disable=True)
    client = groq.Groq()
    client.audio.transcriptions.create(**DUMMY_AUDIO_TRANSCRIPTION_REQUEST)

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1


DUMMY_AUDIO_TRANSLATION_REQUEST = {
    "file": ("test_audio.m4a", BINARY_CONTENT),
    "model": "whisper-large-v3",
}

DUMMY_AUDIO_TRANSLATION_RESPONSE = Translation(text="Test audio", x_groq={"id": "req_test"})


@patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
@patch("groq._client.Groq.post", return_value=DUMMY_AUDIO_TRANSLATION_RESPONSE)
def test_audio_translation_autolog(mock_post):
    mlflow.groq.autolog()
    client = groq.Groq()
    client.audio.translations.create(**DUMMY_AUDIO_TRANSLATION_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Translations"
    assert span.span_type == SpanType.LLM
    assert span.inputs["file"][0] == DUMMY_AUDIO_TRANSLATION_REQUEST["file"][0]
    assert span.inputs["file"][1] == str(DUMMY_AUDIO_TRANSLATION_REQUEST["file"][1])
    assert span.inputs["model"] == DUMMY_AUDIO_TRANSLATION_REQUEST["model"]
    assert span.outputs == DUMMY_AUDIO_TRANSLATION_RESPONSE.to_dict()

    mlflow.groq.autolog(disable=True)
    client = groq.Groq()
    client.audio.translations.create(**DUMMY_AUDIO_TRANSLATION_REQUEST)

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1
