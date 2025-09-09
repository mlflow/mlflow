from types import SimpleNamespace
from unittest.mock import patch

import pytest
import smolagents
from packaging.version import Version

import mlflow
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces

_DUMMY_INPUT = "Explain quantum mechanics in simple terms."
_SMOLAGENTS_VERSION_NEW = Version(smolagents.__version__) >= Version("1.15.0")
MOCK_INFERENCE_CLIENT_MODEL_METHOD = "generate" if _SMOLAGENTS_VERSION_NEW else "__call__"


def clear_autolog_state():
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    for key in AUTOLOGGING_INTEGRATIONS.keys():
        AUTOLOGGING_INTEGRATIONS[key].clear()
    mlflow.utils.import_hooks._post_import_hooks = {}


def test_run_autolog():
    from smolagents import ChatMessage, CodeAgent, InferenceClientModel

    _DUMMY_OUTPUT = ChatMessage(
        role="user",
        content='[{"type": "text", "text": "Explain quantum mechanics in simple terms."}]',
    )
    _DUMMY_OUTPUT.raw = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=18,
            total_tokens=28,
        )
    )

    clear_autolog_state()
    agent = CodeAgent(
        tools=[],
        model=InferenceClientModel(model_id="gpt-3.5-turbo", token="test_id"),
        max_steps=2,
    )
    with patch(
        f"smolagents.InferenceClientModel.{MOCK_INFERENCE_CLIENT_MODEL_METHOD}",
        return_value=_DUMMY_OUTPUT,
    ):
        mlflow.smolagents.autolog()
        agent.run(_DUMMY_INPUT)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    if _SMOLAGENTS_VERSION_NEW:
        # TODO: support this once the new version is stable
        assert len(traces[0].data.spans) > 0
    else:
        assert len(traces[0].data.spans) == 6
        # CodeAgent
        span_0 = traces[0].data.spans[0]
        assert span_0.name == "CodeAgent.run"
        assert span_0.span_type == SpanType.AGENT
        assert span_0.parent_id is None
        assert span_0.inputs == {"task": _DUMMY_INPUT}
        assert span_0.outputs == {
            "_value": '[{"type": "text", "text": "Explain quantum mechanics in simple terms."}]'
        }
        # CodeAgent
        span_1 = traces[0].data.spans[1]
        assert span_1.name == "CodeAgent.step_1"
        assert span_1.span_type == SpanType.AGENT
        assert span_1.parent_id == span_0.span_id
        assert span_1.inputs["memory_step"]["step_number"] == 1
        assert span_1.outputs is None
        # InferenceClientModel
        span_2 = traces[0].data.spans[2]
        assert span_2.name == "InferenceClientModel.call_original_1"
        assert span_2.span_type == SpanType.CHAT_MODEL
        assert span_2.parent_id == span_1.span_id
        assert span_2.inputs is not None
        assert span_2.outputs is not None
        # CodeAgent
        span_3 = traces[0].data.spans[3]
        assert span_3.name == "CodeAgent.step_2"
        assert span_3.span_type == SpanType.AGENT
        assert span_3.parent_id == span_0.span_id
        assert span_3.inputs is not None
        assert span_3.outputs is None
        # InferenceClientModel
        span_4 = traces[0].data.spans[4]
        assert span_4.name == "InferenceClientModel.call_original_2"
        assert span_4.span_type == SpanType.CHAT_MODEL
        assert span_4.parent_id == span_3.span_id
        assert span_4.inputs is not None
        assert span_4.outputs is not None
        # InferenceClientModel
        span_5 = traces[0].data.spans[5]
        assert span_5.name == "InferenceClientModel.call_original_3"
        assert span_5.span_type == SpanType.CHAT_MODEL
        assert span_5.parent_id == span_0.span_id
        assert span_5.inputs is not None
        assert span_5.outputs is not None

        assert span_2.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
            "input_tokens": 10,
            "output_tokens": 18,
            "total_tokens": 28,
        }
        assert span_4.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
            "input_tokens": 10,
            "output_tokens": 18,
            "total_tokens": 28,
        }
        assert span_5.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
            "input_tokens": 10,
            "output_tokens": 18,
            "total_tokens": 28,
        }

        assert traces[0].info.token_usage == {
            "input_tokens": 30,
            "output_tokens": 54,
            "total_tokens": 84,
        }

    clear_autolog_state()


def test_run_failure():
    from smolagents import CodeAgent, InferenceClientModel

    clear_autolog_state()
    mlflow.smolagents.autolog()
    agent = CodeAgent(
        tools=[],
        model=InferenceClientModel(model_id="gpt-3.5-turbo", token="test_id"),
        max_steps=1,
    )
    with patch(
        f"smolagents.InferenceClientModel.{MOCK_INFERENCE_CLIENT_MODEL_METHOD}",
        side_effect=Exception("error"),
    ):
        with pytest.raises(Exception, match="error"):
            agent.run(_DUMMY_INPUT)
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    if _SMOLAGENTS_VERSION_NEW:
        assert len(traces[0].data.spans) > 0
    else:
        assert len(traces[0].data.spans) == 2
        # CodeAgent
        span_0 = traces[0].data.spans[0]
        assert span_0.name == "CodeAgent.run"
        assert span_0.span_type == SpanType.AGENT
        assert span_0.parent_id is None
        assert span_0.inputs == {"task": _DUMMY_INPUT}
        assert span_0.outputs is None
        # InferenceClientModel
        span_1 = traces[0].data.spans[1]
        assert span_1.name == "CodeAgent.step"
        assert span_1.span_type == SpanType.AGENT
        assert span_1.parent_id == span_0.span_id
        assert span_1.inputs is not None
        assert span_1.outputs is None

    clear_autolog_state()


def test_tool_autolog():
    from smolagents import ChatMessage, CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

    _DUMMY_OUTPUT = ChatMessage(
        role="user",
        content='[{"type": "text", "text": "Explain quantum mechanics in simple terms."}]',
    )
    _DUMMY_OUTPUT.raw = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=18,
            total_tokens=28,
        )
    )
    clear_autolog_state()
    agent = CodeAgent(
        tools=[
            DuckDuckGoSearchTool(),
        ],
        model=InferenceClientModel(model_id="gpt-3.5-turbo", token="test_id"),
        max_steps=1,
    )
    with patch(
        f"smolagents.InferenceClientModel.{MOCK_INFERENCE_CLIENT_MODEL_METHOD}",
        return_value=_DUMMY_OUTPUT,
    ):
        mlflow.smolagents.autolog()
        agent.run(_DUMMY_INPUT)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    if _SMOLAGENTS_VERSION_NEW:
        assert len(traces[0].data.spans) > 0
    else:
        assert len(traces[0].data.spans) == 4
        # CodeAgent
        span_0 = traces[0].data.spans[0]
        assert span_0.name == "CodeAgent.run"
        assert span_0.span_type == SpanType.AGENT
        assert span_0.parent_id is None
        assert span_0.inputs is not None
        assert span_0.outputs is not None
        # InferenceClientModel
        span_1 = traces[0].data.spans[1]
        assert span_1.name == "CodeAgent.step"
        assert span_1.span_type == SpanType.AGENT
        assert span_1.parent_id == span_0.span_id
        assert span_1.inputs is not None
        assert span_1.outputs is None
        # InferenceClientModel
        span_2 = traces[0].data.spans[2]
        assert span_2.name == "InferenceClientModel.call_original_1"
        assert span_2.span_type == SpanType.CHAT_MODEL
        assert span_2.parent_id == span_1.span_id
        assert span_2.inputs is not None
        assert span_2.outputs is not None
        # CodeAgent
        span_3 = traces[0].data.spans[3]
        assert span_3.name == "InferenceClientModel.call_original_2"
        assert span_3.span_type == SpanType.CHAT_MODEL
        assert span_3.parent_id == span_0.span_id
        assert span_3.inputs is not None
        assert span_3.outputs is not None

        assert span_2.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
            "input_tokens": 10,
            "output_tokens": 18,
            "total_tokens": 28,
        }
        assert span_3.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
            "input_tokens": 10,
            "output_tokens": 18,
            "total_tokens": 28,
        }

        assert traces[0].info.token_usage == {
            "input_tokens": 20,
            "output_tokens": 36,
            "total_tokens": 56,
        }

    clear_autolog_state()
