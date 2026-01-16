import pytest

from mlflow import MlflowClient, get_experiment_by_name, set_experiment, set_tracking_uri
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DemoFeature, DemoResult
from mlflow.demo.generators.traces import (
    DEMO_TRACE_TYPE_TAG,
    DEMO_VERSION_TAG,
    TracesDemoGenerator,
)


@pytest.fixture
def tracking_uri(tmp_path):
    uri = tmp_path / "mlruns"
    uri.mkdir()
    set_tracking_uri(uri.as_uri())
    yield uri.as_uri()
    set_tracking_uri(None)


def test_generator_attributes():
    generator = TracesDemoGenerator()
    assert generator.name == DemoFeature.TRACES
    assert generator.version == 1


def test_data_exists_false_when_no_experiment(tracking_uri):
    generator = TracesDemoGenerator()
    assert generator._data_exists() is False


def test_data_exists_false_when_experiment_empty(tracking_uri):
    set_experiment(DEMO_EXPERIMENT_NAME)
    generator = TracesDemoGenerator()
    assert generator._data_exists() is False


def test_generate_creates_traces(tracking_uri):
    generator = TracesDemoGenerator()
    result = generator.generate()

    assert isinstance(result, DemoResult)
    assert result.feature == DemoFeature.TRACES
    assert len(result.entity_ids) > 0
    assert "traces" in result.navigation_url


def test_generate_creates_experiment(tracking_uri):
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    assert experiment is not None
    assert experiment.lifecycle_stage == "active"


def test_data_exists_true_after_generate(tracking_uri):
    generator = TracesDemoGenerator()
    assert generator._data_exists() is False

    generator.generate()

    assert generator._data_exists() is True


def test_delete_demo_removes_traces(tracking_uri):
    generator = TracesDemoGenerator()
    generator.generate()
    assert generator._data_exists() is True

    generator.delete_demo()

    assert generator._data_exists() is False


def test_traces_have_expected_structure(tracking_uri):
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
    assert "embed_query" in all_span_names
    assert "retrieve_docs" in all_span_names
    assert "generate_response" in all_span_names


def test_traces_have_version_metadata(tracking_uri):
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=50)

    v1_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_VERSION_TAG) == "v1"]
    v2_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_VERSION_TAG) == "v2"]

    # 2 RAG + 2 agent + 6 session = 10 per version
    assert len(v1_traces) == 10
    assert len(v2_traces) == 10
    assert len(traces) == 20


def test_traces_have_type_metadata(tracking_uri):
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=50)

    baseline = [t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "baseline"]
    improved = [t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "improved"]

    # All v1 traces are baseline, all v2 traces are improved
    assert len(baseline) == 10
    assert len(improved) == 10


def test_is_generated_checks_version(tracking_uri):
    generator = TracesDemoGenerator()
    generator.generate()
    generator.store_version()

    assert generator.is_generated() is True

    TracesDemoGenerator.version = 2
    assert generator.is_generated() is False

    TracesDemoGenerator.version = 1
