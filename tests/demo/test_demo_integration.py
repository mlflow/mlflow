"""Integration tests for the demo data framework.

These tests run against a real MLflow tracking server to verify that demo data
is correctly persisted, retrieved, and cleaned up on version bumps.
"""

from pathlib import Path

import pytest

from mlflow import MlflowClient, set_tracking_uri
from mlflow.demo import generate_all_demos
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DEMO_PROMPT_PREFIX
from mlflow.demo.data import DEMO_PROMPTS
from mlflow.demo.generators.evaluation import (
    DEMO_DATASET_V1_NAME,
    DEMO_DATASET_V2_NAME,
    EvaluationDemoGenerator,
)
from mlflow.demo.generators.prompts import PromptsDemoGenerator
from mlflow.demo.generators.traces import (
    DEMO_TRACE_TYPE_TAG,
    DEMO_VERSION_TAG,
    TracesDemoGenerator,
)
from mlflow.demo.registry import demo_registry
from mlflow.genai.datasets import search_datasets
from mlflow.genai.prompts import load_prompt, search_prompts
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


@pytest.fixture
def evaluation_generator():
    generator = EvaluationDemoGenerator()
    original_version = generator.version
    yield generator
    EvaluationDemoGenerator.version = original_version


@pytest.fixture
def prompts_generator():
    generator = PromptsDemoGenerator()
    original_version = generator.version
    yield generator
    PromptsDemoGenerator.version = original_version


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
    traces = tracking_server.search_traces(locations=[experiment.experiment_id], max_results=200)

    assert len(traces) == len(result.entity_ids)
    assert len(traces) == 34


def test_traces_have_expected_span_types(tracking_server, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = tracking_server.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces = tracking_server.search_traces(locations=[experiment.experiment_id], max_results=200)

    all_span_names = {span.name for trace in traces for span in trace.data.spans}

    assert "rag_pipeline" in all_span_names
    assert "embed_query" in all_span_names
    assert "retrieve_docs" in all_span_names
    assert "generate_response" in all_span_names
    assert "agent" in all_span_names
    assert "chat_agent" in all_span_names
    assert "prompt_chain" in all_span_names
    assert "render_prompt" in all_span_names


def test_traces_session_metadata(tracking_server, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = tracking_server.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces = tracking_server.search_traces(locations=[experiment.experiment_id], max_results=200)

    session_traces = [t for t in traces if t.info.trace_metadata.get("mlflow.trace.session")]
    assert len(session_traces) == 14

    session_ids = {t.info.trace_metadata.get("mlflow.trace.session") for t in session_traces}
    assert len(session_ids) == 6


def test_traces_version_metadata(tracking_server, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = tracking_server.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces = tracking_server.search_traces(locations=[experiment.experiment_id], max_results=200)

    v1_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_VERSION_TAG) == "v1"]
    v2_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_VERSION_TAG) == "v2"]

    assert len(v1_traces) == 17
    assert len(v2_traces) == 17


def test_traces_type_metadata(tracking_server, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = tracking_server.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces = tracking_server.search_traces(locations=[experiment.experiment_id], max_results=200)

    rag_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "rag"]
    agent_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "agent"]
    prompt_traces = [
        t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "prompt"
    ]
    session_traces = [
        t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "session"
    ]

    assert len(rag_traces) == 4
    assert len(agent_traces) == 4
    assert len(prompt_traces) == 12
    assert len(session_traces) == 14


def test_traces_delete_removes_all(tracking_server, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = tracking_server.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces_before = tracking_server.search_traces(
        locations=[experiment.experiment_id], max_results=200
    )
    assert len(traces_before) == 34

    traces_generator.delete_demo()

    traces_after = tracking_server.search_traces(
        locations=[experiment.experiment_id], max_results=200
    )
    assert len(traces_after) == 0


def test_evaluation_creates_two_datasets(tracking_server, evaluation_generator):
    result = evaluation_generator.generate()
    evaluation_generator.store_version()

    assert len(result.entity_ids) == 2  # Two evaluation run IDs

    experiment = tracking_server.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    v1_datasets = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"name = '{DEMO_DATASET_V1_NAME}'",
        max_results=10,
    )
    v2_datasets = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"name = '{DEMO_DATASET_V2_NAME}'",
        max_results=10,
    )

    assert len(v1_datasets) == 1
    assert len(v2_datasets) == 1
    assert v1_datasets[0].name == DEMO_DATASET_V1_NAME
    assert v2_datasets[0].name == DEMO_DATASET_V2_NAME


def test_evaluation_datasets_have_records(tracking_server, evaluation_generator):
    evaluation_generator.generate()
    evaluation_generator.store_version()

    experiment = tracking_server.get_experiment_by_name(DEMO_EXPERIMENT_NAME)

    v1_datasets = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"name = '{DEMO_DATASET_V1_NAME}'",
        max_results=10,
    )
    v2_datasets = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"name = '{DEMO_DATASET_V2_NAME}'",
        max_results=10,
    )

    assert len(v1_datasets) == 1
    assert len(v2_datasets) == 1

    v1_df = v1_datasets[0].to_df()
    v2_df = v2_datasets[0].to_df()

    # Each dataset has one record per trace (17 traces per version)
    assert len(v1_df) == 17
    assert len(v2_df) == 17


def test_evaluation_delete_removes_datasets(tracking_server, evaluation_generator):
    evaluation_generator.generate()
    evaluation_generator.store_version()

    experiment = tracking_server.get_experiment_by_name(DEMO_EXPERIMENT_NAME)

    datasets_before = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string="name LIKE 'demo-%'",
        max_results=10,
    )
    assert len(datasets_before) == 2

    evaluation_generator.delete_demo()

    datasets_after = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string="name LIKE 'demo-%'",
        max_results=10,
    )
    assert len(datasets_after) == 0


def test_prompts_creates_on_server(tracking_server, prompts_generator):
    result = prompts_generator.generate()
    prompts_generator.store_version()

    prompts = search_prompts(
        filter_string=f"name LIKE '{DEMO_PROMPT_PREFIX}.%'",
        max_results=100,
    )

    assert len(prompts) == len(DEMO_PROMPTS)
    assert any("prompts:" in e for e in result.entity_ids)
    assert any("versions:" in e for e in result.entity_ids)


def test_prompts_have_multiple_versions(tracking_server, prompts_generator):
    prompts_generator.generate()
    prompts_generator.store_version()

    for prompt_def in DEMO_PROMPTS:
        expected_versions = len(prompt_def.versions)
        prompt = load_prompt(prompt_def.name, version=expected_versions)
        assert prompt is not None
        assert prompt.version == expected_versions


def test_prompts_have_production_alias(tracking_server, prompts_generator):
    prompts_generator.generate()
    prompts_generator.store_version()

    for prompt_def in DEMO_PROMPTS:
        for version_num, version_def in enumerate(prompt_def.versions, start=1):
            if "production" in version_def.aliases:
                prompt = load_prompt(f"prompts:/{prompt_def.name}@production")
                assert prompt.version == version_num


def test_prompts_delete_removes_all(tracking_server, prompts_generator):
    prompts_generator.generate()
    prompts_generator.store_version()

    prompts_before = search_prompts(
        filter_string=f"name LIKE '{DEMO_PROMPT_PREFIX}.%'",
        max_results=100,
    )
    assert len(prompts_before) == len(DEMO_PROMPTS)

    prompts_generator.delete_demo()

    prompts_after = search_prompts(
        filter_string=f"name LIKE '{DEMO_PROMPT_PREFIX}.%'",
        max_results=100,
    )
    assert len(prompts_after) == 0
