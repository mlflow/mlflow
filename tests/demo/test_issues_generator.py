import json

import pytest

import mlflow
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DemoFeature
from mlflow.demo.data import ASSESSMENT_TO_ISSUE, ISSUE_EVIDENCE_WALKTHROUGHS
from mlflow.demo.generators.issues import (
    _DEMO_DETECTION_HISTORY,
    IssuesDemoGenerator,
    _build_evidence,
    _pick_span,
)
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


class FakeSpan:
    """Enough of a span for the pickers: an id and a type."""

    def __init__(self, span_id: str, span_type: str | None):
        self.span_id = span_id
        self.span_type = span_type


def test_every_assessment_has_a_walkthrough():
    # Otherwise some demo issues get structured evidence and others a bare rationale.
    assert set(ISSUE_EVIDENCE_WALKTHROUGHS) == set(ASSESSMENT_TO_ISSUE)


@pytest.mark.parametrize("assessment_name", sorted(ISSUE_EVIDENCE_WALKTHROUGHS))
def test_highlights_are_exact_substrings(assessment_name):
    # The UI emphasizes by searching the text, and `IssueEvidenceStep` rejects a paraphrase.
    for step in ISSUE_EVIDENCE_WALKTHROUGHS[assessment_name]:
        for highlight in step["highlights"]:
            assert highlight in step["text"]


def test_pick_span_prefers_the_named_types_in_order():
    spans = [FakeSpan("root", "CHAIN"), FakeSpan("llm", "LLM"), FakeSpan("tool", "TOOL")]

    assert _pick_span(spans, ["TOOL", "LLM"], set()).span_id == "tool"
    assert _pick_span(spans, ["LLM", "TOOL"], set()).span_id == "llm"


def test_pick_span_takes_the_last_match_for_a_response_step():
    # A ReAct trace's earlier LLM spans are tool-call decisions (`content: None`); only the last
    # holds the answer. A response step on an earlier one also precedes the tool output above it.
    spans = [
        FakeSpan("root", "CHAIN"),
        FakeSpan("llm-tool-call", "LLM"),
        FakeSpan("tool", "TOOL"),
        FakeSpan("llm-final", "LLM"),
    ]

    assert _pick_span(spans, ["LLM"], set(), last=True).span_id == "llm-final"
    assert _pick_span(spans, ["LLM"], set()).span_id == "llm-tool-call"


def test_build_evidence_on_a_react_trace_cites_the_tool_then_the_answer():
    # The shape that exposed the bug: the two steps must read in the order they happened.
    spans = [
        FakeSpan("root", "CHAIN"),
        FakeSpan("llm-tool-call", "LLM"),
        FakeSpan("tool", "TOOL"),
        FakeSpan("llm-final", "LLM"),
    ]

    evidence = _build_evidence("correctness", "The total is wrong.", spans)

    assert [step.span_id for step in evidence.steps] == ["tool", "llm-final"]
    assert evidence.primary_span_id == "llm-final"


def test_pick_span_avoids_a_span_an_earlier_step_already_cited():
    # Otherwise two steps describe the same place in the trace.
    spans = [FakeSpan("llm-1", "LLM"), FakeSpan("llm-2", "LLM")]

    assert _pick_span(spans, ["LLM"], {"llm-1"}).span_id == "llm-2"


def test_pick_span_reuses_a_span_when_it_is_the_only_one_of_its_type():
    # Better than citing a span whose type cannot support the claim.
    spans = [FakeSpan("root", "CHAIN"), FakeSpan("llm", "LLM")]

    assert _pick_span(spans, ["LLM"], {"llm"}).span_id == "llm"


def test_pick_span_falls_back_when_no_preferred_type_is_present():
    # A step's preferred type is not in every trace shape; the claim still holds, so it cites
    # what the trace has rather than being dropped.
    spans = [FakeSpan("root", "CHAIN")]

    assert _pick_span(spans, ["RETRIEVER", "EMBEDDING"], set()).span_id == "root"


def test_pick_span_on_a_trace_with_no_spans():
    assert _pick_span([], ["LLM"], set()) is None


def test_build_evidence_cites_distinct_spans_and_keeps_the_rationale():
    spans = [FakeSpan("root", "CHAIN"), FakeSpan("retriever", "RETRIEVER"), FakeSpan("llm", "LLM")]

    evidence = _build_evidence(
        "groundedness", "The answer cites a figure the context lacks.", spans
    )

    # The judge's own rationale: the only part specific to this trace.
    assert evidence.summary == "The answer cites a figure the context lacks."
    assert [step.span_id for step in evidence.steps] == ["retriever", "llm"]
    # The last step names the failure itself, so its span is the one the finding belongs to.
    assert evidence.primary_span_id == "llm"


def test_build_evidence_without_spans_or_a_walkthrough():
    # Both fall back to the bare rationale, which is still worth linking.
    assert _build_evidence("groundedness", "why", []) is None
    assert _build_evidence("not-an-assessment", "why", [FakeSpan("llm", "LLM")]) is None


def test_detection_history_covers_the_states_the_page_must_render():
    triggers = {spec["scheduled"] for spec in _DEMO_DETECTION_HISTORY}
    statuses = {spec["status"] for spec in _DEMO_DETECTION_HISTORY}

    assert triggers == {True, False}
    # Three terminal statuses, and none left RUNNING.
    assert len(statuses) == 3
    assert any(spec["total_traces"] == 0 for spec in _DEMO_DETECTION_HISTORY)
    assert any(spec.get("error") for spec in _DEMO_DETECTION_HISTORY)
    # The window column is what tells the two triggers apart.
    assert all(spec.get("window_days") for spec in _DEMO_DETECTION_HISTORY if spec["scheduled"])


def test_only_the_runs_that_finished_claim_an_analyzed_count():
    # The count is written when the analysis returns, so a run that failed or was killed has
    # none -- and where there is one it cannot exceed what the run was handed.
    for spec in _DEMO_DETECTION_HISTORY:
        analyzed = spec.get("analyzed_traces")
        if spec["status"] == RunStatus.FINISHED:
            assert analyzed is not None
            assert analyzed <= spec["total_traces"]
        else:
            assert analyzed is None


def test_detection_history_windows_stay_within_the_production_cap():
    # A wider window is one the scheduler could never have produced.
    cap_days = STALE_WATERMARK_CAP_MS / (24 * 60 * 60 * 1000)

    assert all(
        spec["window_days"] <= cap_days for spec in _DEMO_DETECTION_HISTORY if spec["scheduled"]
    )


def test_the_widest_window_follows_the_failure_that_held_the_watermark():
    # The wide window exists because the runs before it held the watermark, so it sits after
    # them -- fewer days ago.
    scheduled = [spec for spec in _DEMO_DETECTION_HISTORY if spec["scheduled"]]
    widest = min(scheduled, key=lambda spec: -spec["window_days"])
    failure = next(spec for spec in scheduled if spec.get("error"))

    assert widest["days_ago"] < failure["days_ago"]
    assert widest["window_days"] > failure["window_days"]


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
