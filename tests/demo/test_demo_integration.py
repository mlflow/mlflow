"""Integration tests for the demo data framework.

These tests run against a real MLflow tracking server to verify that demo data
is correctly persisted, retrieved, and cleaned up on version bumps.
"""

from pathlib import Path

import pytest

from mlflow import MlflowClient, set_tracking_uri
from mlflow.demo import generate_all_demos
from mlflow.demo.base import DEMO_EXPERIMENT_NAME
from mlflow.demo.generators.traces import TracesDemoGenerator
from mlflow.demo.registry import demo_registry
from mlflow.server import handlers
from mlflow.server.fastapi_app import app
from mlflow.server.handlers import initialize_backend_stores

from tests.helper_functions import get_safe_port
from tests.tracking.integration_test_utils import ServerThread


@pytest.fixture
def tracking_server(tmp_path: Path):
    backend_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    handlers._tracking_store = None
    handlers._model_registry_store = None
    initialize_backend_stores(backend_uri, default_artifact_root=tmp_path.as_uri())

    with ServerThread(app, get_safe_port()) as url:
        set_tracking_uri(url)
        yield MlflowClient(url)
        set_tracking_uri(None)


@pytest.fixture
def traces_generator():
    generator = TracesDemoGenerator()
    original_version = generator.version
    yield generator
    TracesDemoGenerator.version = original_version


def test_generate_all_demos_generates_all_registered(tracking_server):
    results = generate_all_demos()

    registered_names = set(demo_registry.list_generators())
    generated_names = {r.feature for r in results}
    assert generated_names == registered_names


def test_generate_all_demos_is_idempotent(tracking_server):
    results_first = generate_all_demos()
    assert len(results_first) > 0

    results_second = generate_all_demos()
    assert len(results_second) == 0


def test_generate_all_demos_creates_experiment(tracking_server):
    generate_all_demos()

    experiment = tracking_server.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    assert experiment is not None
    assert experiment.lifecycle_stage == "active"


def test_version_mismatch_triggers_cleanup_and_regeneration(tracking_server, traces_generator):
    result = traces_generator.generate()
    traces_generator.store_version()
    original_entity_count = len(result.entity_ids)

    TracesDemoGenerator.version = traces_generator.version + 1

    assert traces_generator.is_generated() is False

    result = traces_generator.generate()
    traces_generator.store_version()

    assert len(result.entity_ids) == original_entity_count
    assert traces_generator.is_generated() is True


def test_traces_creates_on_server(tracking_server, traces_generator):
    result = traces_generator.generate()
    traces_generator.store_version()

    experiment = tracking_server.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces = tracking_server.search_traces(locations=[experiment.experiment_id], max_results=100)

    assert len(traces) == len(result.entity_ids)
    assert len(traces) == 35


def test_traces_have_expected_span_types(tracking_server, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = tracking_server.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces = tracking_server.search_traces(locations=[experiment.experiment_id], max_results=100)

    all_span_names = {span.name for trace in traces for span in trace.data.spans}

    assert "rag_pipeline" in all_span_names
    assert "embed_query" in all_span_names
    assert "retrieve_documents" in all_span_names
    assert "generate_response" in all_span_names
    assert "agent" in all_span_names
    assert "chat" in all_span_names


def test_traces_session_metadata(tracking_server, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = tracking_server.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces = tracking_server.search_traces(locations=[experiment.experiment_id], max_results=100)

    session_traces = [t for t in traces if t.info.trace_metadata.get("mlflow.trace.session")]
    assert len(session_traces) == 20

    session_ids = {t.info.trace_metadata.get("mlflow.trace.session") for t in session_traces}
    assert len(session_ids) == 5


def test_traces_delete_removes_all(tracking_server, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = tracking_server.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces_before = tracking_server.search_traces(
        locations=[experiment.experiment_id], max_results=100
    )
    assert len(traces_before) == 35

    traces_generator.delete_demo()

    traces_after = tracking_server.search_traces(
        locations=[experiment.experiment_id], max_results=100
    )
    assert len(traces_after) == 0
