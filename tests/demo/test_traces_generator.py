import pytest

from mlflow import MlflowClient, get_experiment_by_name, set_experiment
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DemoFeature, DemoResult
from mlflow.demo.generators.traces import (
    DEMO_TRACE_TYPE_TAG,
    DEMO_VERSION_TAG,
    TracesDemoGenerator,
)


@pytest.fixture
def traces_generator():
    generator = TracesDemoGenerator()
    original_version = generator.version
    yield generator
    TracesDemoGenerator.version = original_version


def test_generator_attributes():
    generator = TracesDemoGenerator()
    assert generator.name == DemoFeature.TRACES
    assert generator.version == 1


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
    assert "traces" in result.navigation_url


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
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=50)

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
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=50)

    v1_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_VERSION_TAG) == "v1"]
    v2_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_VERSION_TAG) == "v2"]

    # 2 RAG + 2 agent + 6 prompt + 7 session = 17 per version
    assert len(v1_traces) == 17
    assert len(v2_traces) == 17
    assert len(traces) == 34


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
