import json

import pytest

import mlflow
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DemoFeature
from mlflow.demo.data import ASSESSMENT_TO_ISSUE, ISSUE_EVIDENCE_WALKTHROUGHS
from mlflow.demo.generators.issues import _DEMO_DETECTION_HISTORY, IssuesDemoGenerator
from mlflow.demo.generators.traces import DEMO_VERSION_TAG
from mlflow.entities import SpanType
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.issue_evidence import ISSUE_EVIDENCE_METADATA_KEY
from mlflow.entities.run_status import RunStatus
from mlflow.genai.discovery.checkpoint import STALE_WATERMARK_CAP_MS
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.mlflow_tags import (
    MLFLOW_ISSUE_DETECTION_ERROR,
    MLFLOW_ISSUE_DETECTION_SCHEDULE,
    MLFLOW_ISSUE_DETECTION_TRIGGER,
    MLFLOW_ISSUE_DETECTION_WINDOW_END,
    MLFLOW_ISSUE_DETECTION_WINDOW_START,
    MLFLOW_RUN_TYPE,
    MLFLOW_RUN_TYPE_ISSUE_DETECTION,
)


def test_every_assessment_has_a_walkthrough():
    # Otherwise some demo issues get structured evidence and others a bare rationale.
    assert set(ISSUE_EVIDENCE_WALKTHROUGHS) == set(ASSESSMENT_TO_ISSUE)


@pytest.mark.parametrize("assessment_name", sorted(ISSUE_EVIDENCE_WALKTHROUGHS))
def test_highlights_are_exact_substrings(assessment_name):
    # The UI emphasizes by searching the text, and `IssueEvidenceStep` rejects a paraphrase.
    for step in ISSUE_EVIDENCE_WALKTHROUGHS[assessment_name]:
        for highlight in step["highlights"]:
            assert highlight in step["text"]


def test_detection_history_is_data_the_page_can_render():
    # The history is a hand-written fixture, so what is worth pinning is only what an edit to it
    # could quietly break: that it still covers every state the page renders, that the analyzed
    # count follows the production rule (written when the analysis returns, never more than the
    # run was handed), and that no window is wider than the scheduler could have produced.
    cap_days = STALE_WATERMARK_CAP_MS / (24 * 60 * 60 * 1000)

    assert {spec["scheduled"] for spec in _DEMO_DETECTION_HISTORY} == {True, False}
    assert len({spec["status"] for spec in _DEMO_DETECTION_HISTORY}) == 3
    assert any(spec["total_traces"] == 0 for spec in _DEMO_DETECTION_HISTORY)
    assert any(spec.get("error") for spec in _DEMO_DETECTION_HISTORY)

    for spec in _DEMO_DETECTION_HISTORY:
        analyzed = spec.get("analyzed_traces")
        if spec["status"] == RunStatus.FINISHED:
            assert analyzed is not None
            assert analyzed <= spec["total_traces"]
        else:
            assert analyzed is None
        if spec["scheduled"]:
            assert 0 < spec["window_days"] <= cap_days


@pytest.fixture
def demo_experiment_with_failing_traces():
    """Demo-shaped traces: a v1 chain with retriever and LLM children, each failing a judge."""
    mlflow.set_experiment(DEMO_EXPERIMENT_NAME)
    trace_ids = []
    for index in range(2):
        root = mlflow.start_span_no_context(
            name="rag_pipeline",
            span_type=SpanType.CHAIN,
            inputs={"question": f"question {index}"},
            metadata={DEMO_VERSION_TAG: "v1"},
        )
        retriever = mlflow.start_span_no_context(
            name="retrieve_docs", span_type=SpanType.RETRIEVER, parent_span=root
        )
        retriever.end()
        llm = mlflow.start_span_no_context(
            name="generate_answer", span_type=SpanType.LLM, parent_span=root
        )
        llm.end()
        root.end()
        trace_ids.append(root.trace_id)

    for trace_id in trace_ids:
        mlflow.log_feedback(
            trace_id=trace_id,
            name="groundedness",
            value="no",
            rationale="The answer asserts a figure the retrieved context does not contain.",
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id="judges/groundedness",
            ),
        )

    experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    return experiment.experiment_id


def _detection_runs(experiment_id: str):
    return _get_store().search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.`{MLFLOW_RUN_TYPE}` = '{MLFLOW_RUN_TYPE_ISSUE_DETECTION}'",
        run_view_type=mlflow.entities.ViewType.ALL,
        max_results=50,
    )


def test_generator_attributes():
    generator = IssuesDemoGenerator()

    assert generator.name == DemoFeature.ISSUES


def test_generate_attaches_evidence_to_every_issue_reference(demo_experiment_with_failing_traces):
    experiment_id = demo_experiment_with_failing_traces

    IssuesDemoGenerator().generate()

    issues = _get_store().search_issues(experiment_id=experiment_id)
    assert issues
    traces = mlflow.search_traces(locations=[experiment_id], return_type="list", flush=True)
    # An issue reference carries the issue's *id* as its assessment name, not the issue's name.
    issue_ids = {issue.issue_id for issue in issues}
    references = [
        assessment
        for trace in traces
        for assessment in (trace.info.assessments or [])
        if assessment.name in issue_ids
    ]
    assert references
    for reference in references:
        payload = json.loads((reference.metadata or {})[ISSUE_EVIDENCE_METADATA_KEY])
        steps = payload["steps"]
        assert len(steps) == len(ISSUE_EVIDENCE_WALKTHROUGHS["groundedness"])
        # The reference opens on the span the walkthrough ends at.
        assert reference.span_id == steps[-1]["span_id"]
        for step in steps:
            assert all(highlight in step["text"] for highlight in step["highlights"])


def test_generate_records_the_run_history_the_page_lists(demo_experiment_with_failing_traces):
    experiment_id = demo_experiment_with_failing_traces

    IssuesDemoGenerator().generate()

    runs = _detection_runs(experiment_id)
    # The run that filed the issues, plus the history behind it.
    assert len(runs) == len(_DEMO_DETECTION_HISTORY) + 1
    scheduled = [run for run in runs if run.data.tags.get(MLFLOW_ISSUE_DETECTION_TRIGGER)]
    assert scheduled
    for run in scheduled:
        # Both bounds, or the page cannot name the window at all.
        assert int(run.data.tags[MLFLOW_ISSUE_DETECTION_WINDOW_START]) < int(
            run.data.tags[MLFLOW_ISSUE_DETECTION_WINDOW_END]
        )
    assert any(run.data.tags.get(MLFLOW_ISSUE_DETECTION_ERROR) for run in runs)
    # The column reads the analyzed count, so a finished run has to carry one and one that
    # stopped short must not -- a blank cell is the honest answer there.
    for run in runs:
        analyzed = run.data.tags.get("total_traces_analyzed")
        if run.info.status == "FINISHED":
            assert analyzed is not None
            assert int(analyzed) <= int(run.data.tags["total_traces"])
        else:
            assert analyzed is None
    assert {run.info.status for run in runs} >= {"FINISHED", "FAILED", "KILLED"}
    # The page polls while a loaded run is unsettled, so none may be left RUNNING.
    assert all(run.info.end_time for run in runs)


def test_generate_stores_the_schedule_turned_off(demo_experiment_with_failing_traces):
    experiment_id = demo_experiment_with_failing_traces

    IssuesDemoGenerator().generate()

    schedule = json.loads(
        _get_store().get_experiment(experiment_id).tags[MLFLOW_ISSUE_DETECTION_SCHEDULE]
    )
    # Paused, so it is never dispatched -- but still what gives the windows a timezone.
    assert schedule["paused"] is True
    assert schedule["timezone"]
