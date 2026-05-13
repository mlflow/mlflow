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
    DEMO_DATASET_BASELINE_SESSION_NAME,
    DEMO_DATASET_IMPROVED_SESSION_NAME,
    DEMO_DATASET_TRACE_LEVEL_NAME,
    EvaluationDemoGenerator,
)
from mlflow.demo.generators.issues import (
    DEMO_ISSUE_DETECTION_RUN_NAME,
    IssuesDemoGenerator,
)
from mlflow.demo.generators.judges import DEMO_JUDGE_PREFIX, JudgesDemoGenerator
from mlflow.demo.generators.prompts import PromptsDemoGenerator
from mlflow.demo.generators.traces import (
    DEMO_END_TIME_TAG,
    DEMO_START_TIME_TAG,
    DEMO_TRACE_TYPE_TAG,
    DEMO_VERSION_TAG,
    TracesDemoGenerator,
)
from mlflow.demo.registry import demo_registry
from mlflow.entities.issue import IssueStatus
from mlflow.genai.datasets import search_datasets
from mlflow.genai.prompts import load_prompt, search_prompts
from mlflow.genai.scorers.registry import list_scorers
from mlflow.server import BACKEND_STORE_URI_ENV_VAR
from mlflow.server import handlers as mlflow_handlers
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE, MLFLOW_RUN_TYPE_ISSUE_DETECTION


@pytest.fixture
def client(db_uri: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Point the tracking URI directly at the SQLite database to avoid HTTP
    # overhead during data generation (3,000+ requests per test run).
    monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, db_uri)
    set_tracking_uri(db_uri)
    yield MlflowClient(db_uri)
    set_tracking_uri(None)
    # Clear the global job store to avoid stale database connections across tests
    mlflow_handlers._job_store = None


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


@pytest.fixture
def judges_generator():
    generator = JudgesDemoGenerator()
    original_version = generator.version
    yield generator
    JudgesDemoGenerator.version = original_version


@pytest.fixture
def issues_generator():
    generator = IssuesDemoGenerator()
    original_version = generator.version
    yield generator
    IssuesDemoGenerator.version = original_version


@pytest.fixture
def issues_prerequisites(traces_generator, judges_generator, evaluation_generator):
    traces_generator.generate()
    traces_generator.store_version()
    judges_generator.generate()
    judges_generator.store_version()
    evaluation_generator.generate()
    evaluation_generator.store_version()


def test_generate_all_demos_generates_all_registered(client):
    results = generate_all_demos()

    registered_names = set(demo_registry.list_generators())
    generated_names = {r.feature for r in results}
    assert generated_names == registered_names


def test_generate_all_demos_is_idempotent(client):
    results_first = generate_all_demos()
    assert len(results_first) > 0

    results_second = generate_all_demos()
    assert len(results_second) == 0


def test_generate_all_demos_creates_experiment(client):
    generate_all_demos()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    assert experiment is not None
    assert experiment.lifecycle_stage == "active"


def test_version_mismatch_triggers_cleanup_and_regeneration(client, traces_generator):
    result = traces_generator.generate()
    traces_generator.store_version()
    original_entity_count = len(result.entity_ids)

    TracesDemoGenerator.version = traces_generator.version + 1

    assert traces_generator.is_generated() is False

    result = traces_generator.generate()
    traces_generator.store_version()

    assert len(result.entity_ids) == original_entity_count
    assert traces_generator.is_generated() is True


def test_traces_creates_on_server(client, traces_generator):
    result = traces_generator.generate()
    traces_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=200)

    assert len(traces) == len(result.entity_ids)
    assert len(traces) == 42


def test_traces_have_expected_span_types(client, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=200)

    all_span_names = {span.name for trace in traces for span in trace.data.spans}

    assert "rag_pipeline" in all_span_names
    assert "embed_query" in all_span_names
    assert "retrieve_docs" in all_span_names
    assert "generate_response" in all_span_names
    assert "agent" in all_span_names
    assert "chat_agent" in all_span_names
    assert "prompt_chain" in all_span_names
    assert "render_prompt" in all_span_names
    assert "vision_analysis" in all_span_names
    assert "image_generation" in all_span_names
    assert "audio_transcription" in all_span_names
    assert "text_to_speech" in all_span_names


def test_traces_session_metadata(client, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=200)

    session_traces = [t for t in traces if t.info.trace_metadata.get("mlflow.trace.session")]
    assert len(session_traces) == 14

    session_ids = {t.info.trace_metadata.get("mlflow.trace.session") for t in session_traces}
    assert len(session_ids) == 6


def test_traces_version_metadata(client, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=200)

    v1_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_VERSION_TAG) == "v1"]
    v2_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_VERSION_TAG) == "v2"]

    assert len(v1_traces) == 21
    assert len(v2_traces) == 21


def test_traces_type_metadata(client, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=200)

    rag_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "rag"]
    agent_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "agent"]
    prompt_traces = [
        t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "prompt"
    ]
    session_traces = [
        t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "session"
    ]
    multimodal_traces = [
        t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "multimodal"
    ]

    assert len(rag_traces) == 4
    assert len(agent_traces) == 4
    assert len(prompt_traces) == 12
    assert len(session_traces) == 14
    assert len(multimodal_traces) == 8


def test_traces_creates_time_range_tags(client, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    tags = experiment.tags

    assert DEMO_START_TIME_TAG in tags
    assert DEMO_END_TIME_TAG in tags


def test_traces_delete_removes_all(client, traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    traces_before = client.search_traces(locations=[experiment.experiment_id], max_results=200)
    assert len(traces_before) == 42

    traces_generator.delete_demo()

    traces_after = client.search_traces(locations=[experiment.experiment_id], max_results=200)
    assert len(traces_after) == 0


def test_evaluation_creates_three_datasets(client, evaluation_generator):
    result = evaluation_generator.generate()
    evaluation_generator.store_version()

    assert len(result.entity_ids) == 3  # Three evaluation run IDs

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    trace_level_datasets = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"name = '{DEMO_DATASET_TRACE_LEVEL_NAME}'",
        max_results=10,
    )
    baseline_session_datasets = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"name = '{DEMO_DATASET_BASELINE_SESSION_NAME}'",
        max_results=10,
    )
    improved_session_datasets = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"name = '{DEMO_DATASET_IMPROVED_SESSION_NAME}'",
        max_results=10,
    )

    assert len(trace_level_datasets) == 1
    assert len(baseline_session_datasets) == 1
    assert len(improved_session_datasets) == 1
    assert trace_level_datasets[0].name == DEMO_DATASET_TRACE_LEVEL_NAME
    assert baseline_session_datasets[0].name == DEMO_DATASET_BASELINE_SESSION_NAME
    assert improved_session_datasets[0].name == DEMO_DATASET_IMPROVED_SESSION_NAME


def test_evaluation_datasets_have_records(client, evaluation_generator):
    evaluation_generator.generate()
    evaluation_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)

    trace_level_datasets = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"name = '{DEMO_DATASET_TRACE_LEVEL_NAME}'",
        max_results=10,
    )
    baseline_session_datasets = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"name = '{DEMO_DATASET_BASELINE_SESSION_NAME}'",
        max_results=10,
    )
    improved_session_datasets = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"name = '{DEMO_DATASET_IMPROVED_SESSION_NAME}'",
        max_results=10,
    )

    assert len(trace_level_datasets) == 1
    assert len(baseline_session_datasets) == 1
    assert len(improved_session_datasets) == 1

    trace_level_df = trace_level_datasets[0].to_df()
    baseline_session_df = baseline_session_datasets[0].to_df()
    improved_session_df = improved_session_datasets[0].to_df()

    # Trace-level dataset deduplicates by SHA-256 hash of JSON-serialized inputs.
    # v2 records overwrite v1 when inputs are identical. Multimodal traces with
    # auto-extracted attachments get unique mlflow-attachment:// UUIDs per version,
    # so vision/audio input traces don't dedup (2 rows each), while text-only
    # input traces (image_gen, tts) do dedup (1 row each).
    # 10 text traces + 2 multimodal deduped + 4 multimodal not deduped = 16
    assert len(trace_level_df) == 16
    # Session datasets have 7 traces each (v1 and v2)
    assert len(baseline_session_df) == 7
    assert len(improved_session_df) == 7


def test_evaluation_delete_removes_datasets(client, evaluation_generator):
    evaluation_generator.generate()
    evaluation_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)

    datasets_before = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string="name LIKE 'demo-%'",
        max_results=10,
    )
    assert len(datasets_before) == 3

    evaluation_generator.delete_demo()

    datasets_after = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string="name LIKE 'demo-%'",
        max_results=10,
    )
    assert len(datasets_after) == 0


def test_prompts_creates_on_server(client, prompts_generator):
    result = prompts_generator.generate()
    prompts_generator.store_version()

    prompts = search_prompts(
        filter_string=f"name LIKE '{DEMO_PROMPT_PREFIX}.%'",
        max_results=100,
    )

    assert len(prompts) == len(DEMO_PROMPTS)
    assert any("prompts:" in e for e in result.entity_ids)
    assert any("versions:" in e for e in result.entity_ids)


def test_prompts_have_multiple_versions(client, prompts_generator):
    prompts_generator.generate()
    prompts_generator.store_version()

    for prompt_def in DEMO_PROMPTS:
        expected_versions = len(prompt_def.versions)
        prompt = load_prompt(prompt_def.name, version=expected_versions)
        assert prompt is not None
        assert prompt.version == expected_versions


def test_prompts_have_production_alias(client, prompts_generator):
    prompts_generator.generate()
    prompts_generator.store_version()

    for prompt_def in DEMO_PROMPTS:
        for version_num, version_def in enumerate(prompt_def.versions, start=1):
            if "production" in version_def.aliases:
                prompt = load_prompt(f"prompts:/{prompt_def.name}@production")
                assert prompt.version == version_num


def test_prompts_delete_removes_all(client, prompts_generator):
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


def test_judges_creates_on_server(client, judges_generator):
    result = judges_generator.generate()
    judges_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    scorers = list_scorers(experiment_id=experiment.experiment_id)
    demo_judges = [s for s in scorers if s.name.startswith(DEMO_JUDGE_PREFIX)]

    assert len(demo_judges) == 4
    assert "judges:4" in result.entity_ids


def test_judges_have_expected_names(client, judges_generator):
    judges_generator.generate()
    judges_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    scorers = list_scorers(experiment_id=experiment.experiment_id)
    demo_judges = [s for s in scorers if s.name.startswith(DEMO_JUDGE_PREFIX)]

    judge_names = {s.name for s in demo_judges}
    expected_names = {
        f"{DEMO_JUDGE_PREFIX}.relevance",
        f"{DEMO_JUDGE_PREFIX}.correctness",
        f"{DEMO_JUDGE_PREFIX}.groundedness",
        f"{DEMO_JUDGE_PREFIX}.safety",
    }
    assert judge_names == expected_names


def test_judges_delete_removes_all(client, judges_generator):
    judges_generator.generate()
    judges_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)

    scorers_before = list_scorers(experiment_id=experiment.experiment_id)
    demo_judges_before = [s for s in scorers_before if s.name.startswith(DEMO_JUDGE_PREFIX)]
    assert len(demo_judges_before) == 4

    judges_generator.delete_demo()

    scorers_after = list_scorers(experiment_id=experiment.experiment_id)
    demo_judges_after = [s for s in scorers_after if s.name.startswith(DEMO_JUDGE_PREFIX)]
    assert len(demo_judges_after) == 0


def test_issues_creates_on_server(client, issues_prerequisites, issues_generator):
    result = issues_generator.generate()
    issues_generator.store_version()

    store = _get_store()
    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    issues = store.search_issues(
        experiment_id=experiment.experiment_id,
        max_results=1000,
    )
    demo_issues = [i for i in issues if i.created_by == "demo"]

    assert len(demo_issues) > 0
    assert len(demo_issues) == len(result.entity_ids)


def test_issues_creates_detection_run(client, issues_prerequisites, issues_generator):
    issues_generator.generate()
    issues_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`{MLFLOW_RUN_TYPE}` = '{MLFLOW_RUN_TYPE_ISSUE_DETECTION}'",
        max_results=100,
    )

    assert len(runs) == 1
    assert runs[0].info.run_name == DEMO_ISSUE_DETECTION_RUN_NAME


def test_issues_has_result_tags(client, issues_prerequisites, issues_generator):
    issues_generator.generate()
    issues_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`{MLFLOW_RUN_TYPE}` = '{MLFLOW_RUN_TYPE_ISSUE_DETECTION}'",
        max_results=100,
    )

    assert len(runs) == 1
    run = runs[0]
    assert "mlflow.issueDetection.result.issues" in run.data.tags
    assert "mlflow.issueDetection.result.totalTracesAnalyzed" in run.data.tags
    assert int(run.data.tags["mlflow.issueDetection.result.issues"]) > 0


def test_issues_linked_to_traces(client, issues_prerequisites, issues_generator):
    issues_generator.generate()
    issues_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`{MLFLOW_RUN_TYPE}` = '{MLFLOW_RUN_TYPE_ISSUE_DETECTION}'",
        max_results=100,
    )
    assert len(runs) == 1
    run_id = runs[0].info.run_id

    traces = client.search_traces(
        locations=[experiment.experiment_id],
        filter_string=f"trace.run_id = '{run_id}'",
        max_results=100,
    )
    assert len(traces) > 0


def test_issues_result_metadata(client, issues_prerequisites, issues_generator):
    issues_generator.generate()
    issues_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`{MLFLOW_RUN_TYPE}` = '{MLFLOW_RUN_TYPE_ISSUE_DETECTION}'",
        max_results=100,
    )

    assert len(runs) == 1
    run = runs[0]

    # Check that result metadata is in tags
    issues = run.data.tags.get("mlflow.issueDetection.result.issues")
    total_traces = run.data.tags.get("mlflow.issueDetection.result.totalTracesAnalyzed")
    summary = run.data.tags.get("mlflow.issueDetection.result.summary")

    assert issues is not None
    assert total_traces is not None
    assert int(issues) > 0
    assert int(total_traces) > 0

    # Check that summary is present and has content
    assert summary is not None
    assert len(summary) > 0
    assert "Analyzed" in summary


def test_issues_delete_removes_all(client, issues_prerequisites, issues_generator):
    issues_generator.generate()
    issues_generator.store_version()

    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)

    runs_before = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`{MLFLOW_RUN_TYPE}` = '{MLFLOW_RUN_TYPE_ISSUE_DETECTION}'",
        max_results=100,
    )
    assert len(runs_before) == 1

    issues_generator.delete_demo()

    runs_after = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`{MLFLOW_RUN_TYPE}` = '{MLFLOW_RUN_TYPE_ISSUE_DETECTION}'",
        max_results=100,
    )
    assert len(runs_after) == 0


def test_issues_delete_rejects_pending_demo_issues(client, issues_prerequisites, issues_generator):
    # No delete_issue API exists, so delete_demo() must mark previously created
    # demo issues as REJECTED to prevent duplicates on regeneration.
    issues_generator.generate()
    issues_generator.store_version()

    store = _get_store()
    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)

    issues_before = store.search_issues(experiment_id=experiment.experiment_id, max_results=1000)
    demo_issues_before = [i for i in issues_before if i.created_by == "demo"]
    assert len(demo_issues_before) > 0
    assert all(i.status == IssueStatus.PENDING for i in demo_issues_before)

    issues_generator.delete_demo()

    issues_after = store.search_issues(experiment_id=experiment.experiment_id, max_results=1000)
    demo_issues_after = [i for i in issues_after if i.created_by == "demo"]
    # Same issue records (no delete API), but all now REJECTED.
    assert len(demo_issues_after) == len(demo_issues_before)
    assert all(i.status == IssueStatus.REJECTED for i in demo_issues_after)
