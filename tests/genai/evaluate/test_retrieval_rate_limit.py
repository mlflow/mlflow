import contextvars
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from mlflow.entities import Feedback
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.harness import (
    _compute_eval_scores,
    _make_rate_limiter,
    _parse_rate_limit,
)
from mlflow.genai.evaluation.rate_limiter import RPSRateLimiter, scorer_rate_limit_context
from mlflow.genai.scorers import RetrievalRelevance
from mlflow.genai.scorers.base import scorer

from tests.genai.conftest import databricks_only


class Clock:
    def __init__(self):
        self.now = 0.0
        self.lock = threading.Lock()

    def monotonic(self):
        with self.lock:
            return self.now

    def sleep(self, seconds):
        with self.lock:
            self.now += seconds


class RecordingLimiter(RPSRateLimiter):
    def __init__(self, rps=2.0, adaptive=False):
        self.clock = Clock()
        super().__init__(rps, adaptive=adaptive, clock=self.clock.monotonic, sleep=self.clock.sleep)
        self.admissions = 0
        self.successes = 0
        self.throttles = 0
        self.record_lock = threading.Lock()

    def acquire(self):
        super().acquire()
        with self.record_lock:
            self.admissions += 1

    def report_success(self):
        super().report_success()
        with self.record_lock:
            self.successes += 1

    def report_throttle(self):
        super().report_throttle()
        with self.record_lock:
            self.throttles += 1


@pytest.fixture
def request_limiting_sdk(monkeypatch):
    # Model the optional SDK contract, so MLflow's tests also exercise older
    # installed SDKs. Real SDK transport/retry behavior is covered in its suite.
    sdk = pytest.importorskip("databricks.agents.evals.judges")
    active_limiter = contextvars.ContextVar("test_sdk_request_limiter", default=None)
    state = SimpleNamespace(throttle_once=False)
    lock = threading.Lock()

    @contextmanager
    def use_judge_request_rate_limiter(limiter):
        token = active_limiter.set(limiter)
        try:
            yield
        finally:
            active_limiter.reset(token)

    module = ModuleType("databricks.rag_eval.clients.managedrag.request_limiter")
    module.use_judge_request_rate_limiter = use_judge_request_rate_limiter
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def chunk_relevance(*, request, retrieved_context, assessment_name):
        def score_chunk(chunk):
            limiter = active_limiter.get()
            if limiter is not None:
                limiter.acquire()
                with lock:
                    throttled = state.throttle_once
                    state.throttle_once = False
                if throttled:
                    limiter.report_throttle()
                    limiter.acquire()
                limiter.report_success()
            return Feedback(name=assessment_name, value="yes", rationale=chunk["content"])

        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="TestRetrievalChunk") as pool:
            futures = [
                pool.submit(contextvars.copy_context().run, score_chunk, chunk)
                for chunk in retrieved_context
            ]
            return [future.result() for future in futures]

    monkeypatch.setattr(sdk, "chunk_relevance", chunk_relevance)
    return SimpleNamespace(module=module, active_limiter=active_limiter, state=state)


def make_item(trace):
    return EvalItem(request_id="request", inputs={}, outputs="answer", expectations={}, trace=trace)


@databricks_only
@pytest.mark.parametrize("model", [None, "databricks"])
@pytest.mark.parametrize("sdk_supports_limiting", [False, True])
def test_retrieval_charges_requests_or_falls_back_to_invocations(
    request_limiting_sdk, sample_rag_trace, model, sdk_supports_limiting, monkeypatch
):
    if not sdk_supports_limiting:
        monkeypatch.delattr(request_limiting_sdk.module, "use_judge_request_rate_limiter")
    limiter = RecordingLimiter()
    result = _compute_eval_scores(
        eval_item=make_item(sample_rag_trace),
        scorers=[RetrievalRelevance(model=model)],
        rate_limiter=limiter,
    )

    # The fixture contains two retriever spans with two and one documents.
    chunks = [a for a in result.assessments if a.name == "retrieval_relevance"]
    assert sorted(a.rationale for a in chunks) == ["content_1", "content_2", "content_3"]
    assert all(a.value == "yes" and a.error is None for a in chunks)
    assert len(result.assessments) == 5
    assert limiter.admissions == (3 if sdk_supports_limiting else 1)
    assert limiter.successes == limiter.admissions
    assert limiter.clock.now == (0.5 if sdk_supports_limiting else 0.0)


@databricks_only
def test_retrieval_and_other_scorers_share_one_budget_across_rows(
    request_limiting_sdk, sample_rag_trace
):
    @scorer
    def single_call_scorer(outputs):
        assert request_limiting_sdk.active_limiter.get() is None
        return outputs == "answer"

    limiter = RecordingLimiter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="TestEvaluationRow") as pool:
        futures = [
            pool.submit(
                _compute_eval_scores,
                eval_item=make_item(sample_rag_trace),
                scorers=[RetrievalRelevance(model="databricks"), single_call_scorer],
                rate_limiter=limiter,
            )
            for _ in range(2)
        ]
        for future in futures:
            result = future.result()
            assert len(result.assessments) == 6
            assert all(a.error is None for a in result.assessments)

    assert limiter.admissions == 8  # Two rows times (three documents + one other scorer).
    assert limiter.successes == 8
    assert limiter.clock.now >= 3.0


@databricks_only
def test_adaptive_limiter_sees_chunk_throttle_and_retry(request_limiting_sdk, sample_rag_trace):
    request_limiting_sdk.state.throttle_once = True
    limiter = RecordingLimiter(10.0, adaptive=True)
    result = _compute_eval_scores(
        eval_item=make_item(sample_rag_trace),
        scorers=[RetrievalRelevance(model="databricks")],
        rate_limiter=limiter,
    )
    assert all(a.error is None for a in result.assessments)
    assert limiter.admissions == 4
    assert limiter.successes == 3
    assert limiter.throttles == 1
    assert limiter.current_rps < 10.0


@databricks_only
def test_zero_rate_keeps_request_limiting_disabled(request_limiting_sdk, sample_rag_trace):
    rate, adaptive = _parse_rate_limit("0")
    limiter = _make_rate_limiter(rate, adaptive=adaptive)
    result = _compute_eval_scores(
        eval_item=make_item(sample_rag_trace),
        scorers=[RetrievalRelevance(model="databricks")],
        rate_limiter=limiter,
    )
    assert len(result.assessments) == 5
    assert all(a.error is None for a in result.assessments)
    assert request_limiting_sdk.active_limiter.get() is None


@databricks_only
def test_request_context_is_restored_after_exception(request_limiting_sdk):
    judge = RetrievalRelevance(model="databricks")
    outer = RecordingLimiter()
    inner = RecordingLimiter()
    with scorer_rate_limit_context(judge, outer):
        with pytest.raises(ValueError, match="failed"):
            with scorer_rate_limit_context(judge, inner):
                raise ValueError("failed")
        assert request_limiting_sdk.active_limiter.get() is outer
    assert request_limiting_sdk.active_limiter.get() is None


@pytest.mark.parametrize("rate", ["2", "auto"])
def test_other_retrieval_providers_keep_invocation_limiting(rate):
    rps, adaptive = _parse_rate_limit(rate)
    limiter = RecordingLimiter(rps, adaptive=adaptive)
    judge = RetrievalRelevance(model="openai:/gpt-4.1-mini")
    with scorer_rate_limit_context(judge, limiter) as invocation_limiter:
        invocation_limiter.acquire()
    assert limiter.admissions == 1
