"""
End-to-end tests for the online scoring pipeline against a real SQLite tracking store.

These exercise discovery, sampling, full-trace loading, scoring, assessment persistence and
checkpoint updates together, with scorers that need no model endpoint. They pin down the
durable-watermark guarantees that the unit tests assert piecemeal:

* a trace that is still in progress is neither scored on partial data nor skipped;
* re-running a window never produces a second visible assessment for the same scorer;
* the checkpoint resumes exactly where it stopped, even inside one millisecond.
"""

import json
import time
import uuid
from unittest import mock

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as _OTelResource
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow
from mlflow.entities import Feedback, Span, TraceState
from mlflow.entities.span import create_mlflow_span
from mlflow.environment_variables import (
    MLFLOW_ENABLE_ASYNC_TRACE_LOGGING,
    MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS,
    MLFLOW_ONLINE_SCORING_TRACE_COMPLETION_BUFFER_SECONDS,
)
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers.online.constants import MAX_TRACE_COMPLETION_WAIT_MS
from mlflow.genai.scorers.online.entities import OnlineScorer, OnlineScoringConfig
from mlflow.genai.scorers.online.session_checkpointer import OnlineSessionCheckpointManager
from mlflow.genai.scorers.online.session_processor import OnlineSessionScoringProcessor
from mlflow.genai.scorers.online.trace_checkpointer import OnlineTraceCheckpointManager
from mlflow.genai.scorers.online.trace_processor import OnlineTraceScoringProcessor
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import AssessmentMetadataKey, SpanAttributeKey, TraceMetadataKey
from mlflow.utils.mlflow_tags import (
    MLFLOW_LATEST_ONLINE_SCORING_SESSION_CHECKPOINT,
    MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT,
)

# Judges built with make_judge round-trip through the registry serialization used by the
# scheduler (custom @scorer functions are refused outside Databricks). The model call is stubbed
# in the ``fake_judge_model`` fixture, so no endpoint is needed.
always_yes = make_judge(
    name="always_yes",
    instructions="Answer yes for {{ outputs }}",
    feedback_value_type=str,
    model="openai:/gpt-4",
)
conversation_length = make_judge(
    name="conversation_length",
    instructions="Count the turns in {{ conversation }}",
    feedback_value_type=int,
    model="openai:/gpt-4",
)


@pytest.fixture(autouse=True)
def fake_judge_model():
    """Stand-in for the judge LLM: counts conversation turns, otherwise answers 'yes'."""

    def _invoke(model_uri, prompt, assessment_name, **kwargs):
        user_message = prompt[-1].content
        if assessment_name == "conversation_length":
            # The conversation is rendered as a JSON list of turns in the user message.
            turns = user_message.count('"role": "user"')
            return Feedback(name=assessment_name, value=turns, rationale="counted")
        return Feedback(name=assessment_name, value="yes", rationale="scored")

    with mock.patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model", side_effect=_invoke
    ) as patched:
        yield patched


def make_online_scorer(scorer_obj, experiment_id: str, sample_rate: float = 1.0) -> OnlineScorer:
    return OnlineScorer(
        name=scorer_obj.name,
        serialized_scorer=json.dumps(scorer_obj.model_dump()),
        online_config=OnlineScoringConfig(
            online_scoring_config_id=uuid.uuid4().hex,
            scorer_id=uuid.uuid4().hex,
            sample_rate=sample_rate,
            experiment_id=experiment_id,
        ),
    )


@pytest.fixture
def store(tmp_path, monkeypatch) -> SqlAlchemyStore:
    db_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_root = (tmp_path / "artifacts").as_uri()
    # Export spans synchronously so traces land in the store as soon as spans end.
    monkeypatch.setenv(MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.name, "false")
    # No lag: traces and sessions become eligible as soon as they are written.
    monkeypatch.setenv(MLFLOW_ONLINE_SCORING_TRACE_COMPLETION_BUFFER_SECONDS.name, "0")
    monkeypatch.setenv(MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS.name, "0")
    # The fluent API (used by the pipeline to log assessments) must hit the same database.
    mlflow.set_tracking_uri(db_uri)
    yield SqlAlchemyStore(db_uri, artifact_root)
    mlflow.set_tracking_uri(None)


@pytest.fixture
def experiment_id(store) -> str:
    exp_id = store.create_experiment(f"online-scoring-{uuid.uuid4().hex}")
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def run_trace_scoring(store, experiment_id, online_scorers) -> None:
    OnlineTraceScoringProcessor.create(experiment_id, online_scorers, store).process_traces()


def run_session_scoring(store, experiment_id, online_scorers) -> None:
    OnlineSessionScoringProcessor.create(experiment_id, online_scorers, store).process_sessions()


def online_assessments(store, trace_id: str, scorer_name: str) -> list:
    info = store.get_trace_info(trace_id)
    return [
        a
        for a in info.assessments
        if a.name == scorer_name
        and (
            AssessmentMetadataKey.ONLINE_SCORING_SCORER_NAME in (a.metadata or {})
            or AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID in (a.metadata or {})
        )
    ]


def log_trace(session_id: str | None = None, answer: str = "hello") -> str:
    """Log a complete single-span trace through the client, exactly as an app would."""
    with mlflow.start_span("root") as span:
        span.set_inputs({"question": "hi"})
        span.set_outputs(answer)
        if session_id:
            mlflow.update_current_trace(metadata={TraceMetadataKey.TRACE_SESSION: session_id})
    return span.trace_id


def make_root_span(
    trace_id: str, start_ms: int, otel_trace_num: int, end_ms: int | None = None
) -> Span:
    """Build a finished root span with chosen start/end times, for writing via store.log_spans."""
    end_ms = start_ms + 1 if end_ms is None else end_ms
    context = mock.Mock()
    context.trace_id = otel_trace_num
    context.span_id = 1
    context.is_remote = False
    context.trace_flags = trace_api.TraceFlags(1)
    context.trace_state = trace_api.TraceState()
    otel_span = OTelReadableSpan(
        name="root",
        context=context,
        parent=None,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id),
            "mlflow.spanType": json.dumps("CHAIN"),
            "mlflow.spanInputs": json.dumps({"question": "hi"}),
            "mlflow.spanOutputs": json.dumps("hello"),
        },
        start_time=start_ms * 1_000_000,
        end_time=end_ms * 1_000_000,
        status=trace_api.Status(trace_api.StatusCode.OK),
        resource=_OTelResource.get_empty(),
    )
    return create_mlflow_span(otel_span, trace_id, "CHAIN")


def test_in_progress_trace_is_held_then_scored_exactly_once(store, experiment_id):
    scorers = [make_online_scorer(always_yes, experiment_id)]
    checkpoints = OnlineTraceCheckpointManager(store, experiment_id)

    # A completed trace, followed by a trace whose child span has been exported but whose
    # root span is still open. Incremental span export makes the second trace visible to
    # search_traces with state IN_PROGRESS while its data is still incomplete.
    done_trace_id = log_trace()
    time.sleep(0.005)
    with mlflow.start_span("long-root") as root:
        root.set_inputs({"question": "slow"})
        with mlflow.start_span("child"):
            pass
        running_trace_id = root.trace_id
        assert store.get_trace_info(running_trace_id).state == TraceState.IN_PROGRESS

        run_trace_scoring(store, experiment_id, scorers)

        # The running trace was neither scored on partial data nor skipped: the completed
        # trace was scored and the checkpoint stopped before the running one.
        assert len(online_assessments(store, done_trace_id, "always_yes")) == 1
        assert online_assessments(store, running_trace_id, "always_yes") == []
        held = checkpoints.get_checkpoint()
        assert held is not None
        assert held.timestamp_ms < store.get_trace_info(running_trace_id).timestamp_ms
        root.set_outputs("done")

    # Once the trace completes, the next run scores it and moves on. The completed trace is
    # re-discovered (it shares the held window) but is not scored a second time.
    run_trace_scoring(store, experiment_id, scorers)
    assert len(online_assessments(store, running_trace_id, "always_yes")) == 1
    assert len(online_assessments(store, done_trace_id, "always_yes")) == 1
    advanced = checkpoints.get_checkpoint()
    assert advanced.timestamp_ms >= store.get_trace_info(running_trace_id).timestamp_ms


def test_rerunning_a_window_never_duplicates_trace_assessments(store, experiment_id):
    """Simulates a job retry after a crash that happened before the checkpoint was persisted."""
    scorers = [make_online_scorer(always_yes, experiment_id)]
    trace_ids = [log_trace() for _ in range(3)]

    run_trace_scoring(store, experiment_id, scorers)
    # Roll the checkpoint back as if the previous job died before persisting it.
    store.delete_experiment_tag(experiment_id, MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT)
    run_trace_scoring(store, experiment_id, scorers)

    for trace_id in trace_ids:
        assessments = online_assessments(store, trace_id, "always_yes")
        assert len(assessments) == 1
        assert (
            assessments[0].metadata[AssessmentMetadataKey.ONLINE_SCORING_SCORER_NAME]
            == "always_yes"
        )


def test_checkpoint_resumes_within_same_millisecond_without_skipping(store, experiment_id):
    """
    Five traces share one millisecond and only two fit in a batch. Each batch must pick up
    exactly where the previous one stopped, by (timestamp, trace_id), so every trace is scored
    exactly once.
    """
    scorers = [make_online_scorer(always_yes, experiment_id)]
    checkpoints = OnlineTraceCheckpointManager(store, experiment_id)
    start_ms = int(time.time() * 1000) - 5000
    trace_ids = sorted(f"tr-{uuid.uuid4().hex}" for _ in range(5))
    for i, trace_id in enumerate(trace_ids):
        store.log_spans(experiment_id, [make_root_span(trace_id, start_ms, otel_trace_num=i + 1)])
        assert store.get_trace_info(trace_id).timestamp_ms == start_ms

    persisted = []
    real_process_batch = OnlineTraceScoringProcessor._process_batch

    def spy_process_batch(self):
        result = real_process_batch(self)
        persisted.append(checkpoints.get_checkpoint())
        return result

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("mlflow.genai.scorers.online.trace_processor.MAX_TRACES_PER_JOB", 2)
        mp.setattr(OnlineTraceScoringProcessor, "_process_batch", spy_process_batch)
        # A single job drains the backlog: each batch resumes exactly after the previous one
        # by (end_time, trace_id), so no trace is skipped or scored twice.
        run_trace_scoring(store, experiment_id, scorers)

    assert [cp.trace_id for cp in persisted] == [trace_ids[1], trace_ids[3], None]
    for trace_id in trace_ids:
        assert len(online_assessments(store, trace_id, "always_yes")) == 1, trace_id
    # The last page was not full, so the window end became the checkpoint.
    assert checkpoints.get_checkpoint().trace_id is None
    assert checkpoints.get_checkpoint().timestamp_ms > start_ms


def test_session_rescoring_replaces_rather_than_duplicates(store, experiment_id):
    scorers = [make_online_scorer(conversation_length, experiment_id)]
    checkpoints = OnlineSessionCheckpointManager(store, experiment_id)
    first = log_trace(session_id="sess-1")
    time.sleep(0.005)
    log_trace(session_id="sess-1")

    run_session_scoring(store, experiment_id, scorers)
    assert [a.value for a in online_assessments(store, first, "conversation_length")] == [2]
    assert checkpoints.get_checkpoint() is not None

    # A new turn arrives, making the session "complete" again at a later timestamp. Re-scoring
    # replaces the old result instead of adding a second one.
    time.sleep(0.005)
    log_trace(session_id="sess-1")
    run_session_scoring(store, experiment_id, scorers)
    assert [a.value for a in online_assessments(store, first, "conversation_length")] == [3]

    # A plain re-run of the same window (e.g. a job retry) also leaves exactly one assessment.
    store.delete_experiment_tag(experiment_id, MLFLOW_LATEST_ONLINE_SCORING_SESSION_CHECKPOINT)
    run_session_scoring(store, experiment_id, scorers)
    assert [a.value for a in online_assessments(store, first, "conversation_length")] == [3]


def test_transient_judge_failure_is_retried_without_leaving_an_error(
    store, experiment_id, fake_judge_model
):
    """
    The judge fails for one trace on the first run (e.g. a rate limit). That trace must not get
    an error assessment; the checkpoint stops before it, and the next run scores it, leaving
    exactly one (successful) assessment. Traces after it are scored on the first run and not
    re-scored on the second.
    """
    scorers = [make_online_scorer(always_yes, experiment_id)]
    checkpoints = OnlineTraceCheckpointManager(store, experiment_id)
    first = log_trace()
    time.sleep(0.005)
    flaky = log_trace(answer="flaky-answer")
    time.sleep(0.005)
    last = log_trace()

    real_invoke = fake_judge_model.side_effect
    attempts = {"flaky": 0}

    def _flaky_invoke(model_uri, prompt, assessment_name, **kwargs):
        if "flaky-answer" in prompt[-1].content and attempts["flaky"] == 0:
            attempts["flaky"] += 1
            raise RuntimeError("429 rate limited")
        return real_invoke(model_uri, prompt, assessment_name, **kwargs)

    fake_judge_model.side_effect = _flaky_invoke

    run_trace_scoring(store, experiment_id, scorers)
    assert len(online_assessments(store, first, "always_yes")) == 1
    assert len(online_assessments(store, last, "always_yes")) == 1
    assert store.get_trace_info(flaky).assessments == []  # no error assessment left behind
    assert checkpoints.get_checkpoint().timestamp_ms < store.get_trace_info(flaky).timestamp_ms

    run_trace_scoring(store, experiment_id, scorers)
    flaky_assessments = online_assessments(store, flaky, "always_yes")
    assert len(flaky_assessments) == 1
    assert flaky_assessments[0].error is None
    assert len(online_assessments(store, first, "always_yes")) == 1
    assert len(online_assessments(store, last, "always_yes")) == 1
    assert checkpoints.get_checkpoint().timestamp_ms >= store.get_trace_info(last).timestamp_ms


def make_child_span(
    trace_id: str, start_ms: int, otel_trace_num: int, session_id: str | None = None
) -> Span:
    """A finished child span whose root is still open: exporting it makes the trace visible in
    the store with state IN_PROGRESS and no usable root span.
    """
    parent = mock.Mock()
    parent.trace_id = otel_trace_num
    parent.span_id = 1
    parent.is_remote = False
    parent.trace_flags = trace_api.TraceFlags(1)
    parent.trace_state = trace_api.TraceState()
    context = mock.Mock()
    context.trace_id = otel_trace_num
    context.span_id = 2
    context.is_remote = False
    context.trace_flags = trace_api.TraceFlags(1)
    context.trace_state = trace_api.TraceState()
    attributes = {
        "mlflow.traceRequestId": json.dumps(trace_id),
        "mlflow.spanType": json.dumps("LLM"),
    }
    if session_id:
        attributes[SpanAttributeKey.SESSION_ID] = json.dumps(session_id)
    otel_span = OTelReadableSpan(
        name="child",
        context=context,
        parent=parent,
        attributes=attributes,
        start_time=start_ms * 1_000_000,
        end_time=start_ms * 1_000_000 + 1_000_000,
        status=trace_api.Status(trace_api.StatusCode.OK),
        resource=_OTelResource.get_empty(),
    )
    return create_mlflow_span(otel_span, trace_id, "LLM")


def test_trace_stuck_in_progress_gets_an_error_assessment_before_being_dropped(
    store, experiment_id
):
    """
    A trace whose root span never arrived (client crashed) is in progress for good. Once it is
    older than the completion wait, online scoring must not just log a warning and move on: an
    error assessment is recorded on the trace so its absence from scoring is visible, and only
    then does the checkpoint pass it.
    """
    scorers = [make_online_scorer(always_yes, experiment_id)]
    checkpoints = OnlineTraceCheckpointManager(store, experiment_id)
    stuck_id = f"tr-{uuid.uuid4().hex}"
    stuck_start = int(time.time() * 1000) - MAX_TRACE_COMPLETION_WAIT_MS - 60_000
    store.log_spans(experiment_id, [make_child_span(stuck_id, stuck_start, otel_trace_num=7)])
    assert store.get_trace_info(stuck_id).state == TraceState.IN_PROGRESS
    healthy_id = log_trace()

    run_trace_scoring(store, experiment_id, scorers)

    recorded = store.get_trace_info(stuck_id).assessments
    assert [a.name for a in recorded] == ["always_yes"]
    assert recorded[0].error is not None
    assert recorded[0].error.error_code == "TRACE_INCOMPLETE"
    assert recorded[0].metadata[AssessmentMetadataKey.ONLINE_SCORING_SCORER_NAME] == "always_yes"
    assert len(online_assessments(store, healthy_id, "always_yes")) == 1
    assert checkpoints.get_checkpoint().timestamp_ms > stuck_start

    # A re-run does not add a second error assessment.
    store.delete_experiment_tag(experiment_id, MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT)
    run_trace_scoring(store, experiment_id, scorers)
    assert len(store.get_trace_info(stuck_id).assessments) == 1


def test_session_with_only_a_stuck_trace_gets_an_error_assessment_before_being_dropped(
    store, experiment_id
):
    scorers = [make_online_scorer(conversation_length, experiment_id)]
    checkpoints = OnlineSessionCheckpointManager(store, experiment_id)
    stuck_id = f"tr-{uuid.uuid4().hex}"
    stuck_start = int(time.time() * 1000) - MAX_TRACE_COMPLETION_WAIT_MS - 60_000
    store.log_spans(
        experiment_id,
        [make_child_span(stuck_id, stuck_start, otel_trace_num=9, session_id="sess-stuck")],
    )
    assert store.get_trace_info(stuck_id).state == TraceState.IN_PROGRESS
    assert (
        store.get_trace_info(stuck_id).trace_metadata[TraceMetadataKey.TRACE_SESSION]
        == "sess-stuck"
    )

    run_session_scoring(store, experiment_id, scorers)

    recorded = store.get_trace_info(stuck_id).assessments
    assert [a.name for a in recorded] == ["conversation_length"]
    assert recorded[0].error is not None
    assert recorded[0].error.error_code == "TRACE_INCOMPLETE"
    assert recorded[0].metadata[AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID] == "sess-stuck"
    assert checkpoints.get_checkpoint().timestamp_ms > stuck_start


def test_long_running_trace_that_lands_after_the_checkpoint_passed_its_start_is_scored(
    store, experiment_id
):
    """
    A single-span trace is invisible to the store until it finishes. If it runs longer than the
    completion buffer, a checkpoint keyed on *start* time has already moved past it by the time
    it lands and it is silently never scored. Discovery is keyed on *end* time, so the trace is
    found on the first run after it finishes, exactly like a short one.
    """
    scorers = [make_online_scorer(always_yes, experiment_id)]
    checkpoints = OnlineTraceCheckpointManager(store, experiment_id)
    run_started_ms = int(time.time() * 1000)

    # Run 1: only a short trace exists. The long trace is still running (nothing exported yet).
    short_id = log_trace()
    run_trace_scoring(store, experiment_id, scorers)
    assert len(online_assessments(store, short_id, "always_yes")) == 1
    checkpoint_after_run_1 = checkpoints.get_checkpoint().timestamp_ms
    assert checkpoint_after_run_1 >= run_started_ms

    # The long trace now finishes and is exported: it *started* three minutes before run 1
    # (well behind the checkpoint) and ends now.
    long_id = f"tr-{uuid.uuid4().hex}"
    long_start = run_started_ms - 180_000
    long_end = int(time.time() * 1000) + 1
    store.log_spans(
        experiment_id, [make_root_span(long_id, long_start, otel_trace_num=11, end_ms=long_end)]
    )
    info = store.get_trace_info(long_id)
    assert info.state == TraceState.OK
    assert info.timestamp_ms < checkpoint_after_run_1  # behind a start-time checkpoint
    assert info.timestamp_ms + info.execution_time_ms >= checkpoint_after_run_1

    # Run 2 picks it up.
    time.sleep(0.002)
    run_trace_scoring(store, experiment_id, scorers)
    long_assessments = online_assessments(store, long_id, "always_yes")
    assert [(a.name, a.value, a.error) for a in long_assessments] == [("always_yes", "yes", None)]
    # And nothing was double-scored.
    assert len(online_assessments(store, short_id, "always_yes")) == 1


def test_trace_that_completes_after_being_given_up_on_gets_scored_and_placeholder_removed(
    store, experiment_id
):
    """
    A trace stuck in progress past the completion wait gets a TRACE_INCOMPLETE placeholder. If
    its root span eventually arrives, the trace's end time moves ahead of the checkpoint, it is
    re-discovered, scored for real, and the placeholder is removed: the user ends up with one
    correct assessment, not an error.
    """
    scorers = [make_online_scorer(always_yes, experiment_id)]
    stuck_id = f"tr-{uuid.uuid4().hex}"
    stuck_start = int(time.time() * 1000) - MAX_TRACE_COMPLETION_WAIT_MS - 60_000
    store.log_spans(experiment_id, [make_child_span(stuck_id, stuck_start, otel_trace_num=13)])

    run_trace_scoring(store, experiment_id, scorers)
    recorded = store.get_trace_info(stuck_id).assessments
    assert [a.error.error_code for a in recorded] == ["TRACE_INCOMPLETE"]

    # The root span arrives now: the trace completes with an end time of "now".
    root_end = int(time.time() * 1000) + 1
    store.log_spans(
        experiment_id, [make_root_span(stuck_id, stuck_start, otel_trace_num=13, end_ms=root_end)]
    )
    assert store.get_trace_info(stuck_id).state == TraceState.OK

    time.sleep(0.002)
    run_trace_scoring(store, experiment_id, scorers)
    assessments = store.get_trace_info(stuck_id).assessments
    assert [(a.name, a.value, a.error) for a in assessments] == [("always_yes", "yes", None)]
