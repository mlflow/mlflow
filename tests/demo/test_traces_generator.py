import json

import pytest

from mlflow import MlflowClient, get_experiment_by_name, set_experiment
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DemoFeature, DemoResult
from mlflow.demo.generators.traces import (
    DEMO_TRACE_TYPE_TAG,
    DEMO_VERSION_TAG,
    TracesDemoGenerator,
)
from mlflow.tracing.constant import TraceMetadataKey


@pytest.fixture
def traces_generator():
    generator = TracesDemoGenerator()
    original_version = generator.version
    yield generator
    TracesDemoGenerator.version = original_version


def test_generator_attributes():
    generator = TracesDemoGenerator()
    assert generator.name == DemoFeature.TRACES
    assert generator.version == 2


def test_data_exists_false_when_no_experiment():
    generator = TracesDemoGenerator()
    assert generator._data_exists() is False


def test_data_exists_false_when_experiment_empty():
    set_experiment(DEMO_EXPERIMENT_NAME)
    generator = TracesDemoGenerator()
    assert generator._data_exists() is False


def test_generate_creates_traces():
    generator = TracesDemoGenerator()
    result = generator.generate()

    assert isinstance(result, DemoResult)
    assert result.feature == DemoFeature.TRACES
    assert len(result.entity_ids) > 0
    assert "experiments" in result.navigation_url


def test_generate_creates_experiment():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    assert experiment is not None
    assert experiment.lifecycle_stage == "active"


def test_data_exists_true_after_generate():
    generator = TracesDemoGenerator()
    assert generator._data_exists() is False

    generator.generate()

    assert generator._data_exists() is True


def test_delete_demo_removes_traces():
    generator = TracesDemoGenerator()
    generator.generate()
    assert generator._data_exists() is True

    generator.delete_demo()

    assert generator._data_exists() is False


def test_traces_have_expected_structure():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=100)

    assert len(traces) > 0

    all_span_names = set()
    for trace in traces:
        all_span_names.update(span.name for span in trace.data.spans)

    assert "rag_pipeline" in all_span_names
    assert "agent" in all_span_names
    assert "chat_agent" in all_span_names
    assert "prompt_chain" in all_span_names
    assert "render_prompt" in all_span_names
    assert "embed_query" in all_span_names
    assert "retrieve_docs" in all_span_names
    assert "generate_response" in all_span_names


def test_traces_have_version_metadata():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=100)

    v1_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_VERSION_TAG) == "v1"]
    v2_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_VERSION_TAG) == "v2"]

    # 2 RAG + 2 agent + 6 prompt + 4 multimodal + 7 session = 21 per version
    assert len(v1_traces) == 21
    assert len(v2_traces) == 21
    assert len(traces) == 42


def test_traces_have_type_metadata():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=50)

    rag_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "rag"]
    agent_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "agent"]
    prompt_traces = [
        t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "prompt"
    ]
    session_traces = [
        t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "session"
    ]

    # 2 RAG per version = 4 total
    # 2 agent per version = 4 total
    # 6 prompt per version = 12 total
    # 7 session per version = 14 total
    assert len(rag_traces) == 4
    assert len(agent_traces) == 4
    assert len(prompt_traces) == 12
    assert len(session_traces) == 14


def test_is_generated_checks_version(traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    assert traces_generator.is_generated() is True

    TracesDemoGenerator.version = 99
    assert traces_generator.is_generated() is False


def test_agent_and_session_traces_include_chat_messages_attribute():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=100)

    agent_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "agent"]
    session_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "session"]

    assert agent_traces
    assert session_traces

    for trace in agent_traces:
        root_span = next(span for span in trace.data.spans if span.parent_id is None)
        chat_messages = json.loads(root_span.attributes["mlflow.chat.messages"])
        assert [message["role"] for message in chat_messages[:2]] == ["system", "user"]
        assert chat_messages[-1]["role"] == "assistant"
        assert "tool_calls" in chat_messages[2]

    for trace in session_traces:
        root_span = next(span for span in trace.data.spans if span.parent_id is None)
        chat_messages = json.loads(root_span.attributes["mlflow.chat.messages"])
        assert [message["role"] for message in chat_messages] == ["system", "user", "assistant"]
        assert (
            trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION) is not None
        )
