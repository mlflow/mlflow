"""Tests for scorer telemetry behavior, specifically testing that nested scorer calls
skip telemetry recording while top-level calls record telemetry correctly.
"""

import asyncio
import json
import threading
from typing import Callable
from unittest import mock

import pytest
from pydantic import PrivateAttr

from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer
from mlflow.genai.scorers.base import Scorer
from mlflow.telemetry.client import TelemetryClient
from mlflow.telemetry.events import ScorerCallEvent


@scorer
def child_scorer_func(outputs) -> int:
    """Simple child scorer that can be called by parent scorers."""
    return len(outputs)


def parent_scorer_func(child_scorer: Callable[..., int]) -> Scorer:
    """Get a parent scorer that calls into a child scorer and returns the result *2"""

    @scorer
    def scorer_func(outputs) -> Feedback:
        child_result = child_scorer(outputs=outputs)
        return Feedback(name="parent_scorer_func", value=2 * child_result)

    return scorer_func


class ParentScorer(Scorer):
    _child_scorer: Callable[..., int] = PrivateAttr()

    def __init__(self, child_scorer: Callable[..., int], **kwargs):
        super().__init__(**kwargs)
        self._child_scorer = child_scorer

    def __call__(self, outputs) -> Feedback:
        # Call child scorer - this should NOT generate telemetry
        child_result = self._child_scorer(outputs=outputs)
        # child_result is now an int (from decorator scorer), not Feedback
        return Feedback(name=self.name, value=child_result * 2)


class RecursiveScorer(Scorer):
    _max_depth: int = PrivateAttr()

    def __init__(self, max_depth=3, **kwargs):
        super().__init__(**kwargs)
        self._max_depth = max_depth

    def __call__(self, outputs, depth=0) -> Feedback:
        if depth >= self._max_depth:
            return Feedback(name=self.name, value=len(outputs))
        # Recursive call - only first call should generate telemetry
        return self(outputs=outputs, depth=depth + 1)


class GrandparentScorer(Scorer):
    _parent_scorer: Scorer = PrivateAttr()

    def __init__(self, parent_scorer: Scorer, **kwargs):
        super().__init__(**kwargs)
        self._parent_scorer = parent_scorer

    def __call__(self, outputs) -> Feedback:
        parent_result = self._parent_scorer(outputs=outputs)
        return Feedback(name=self.name, value=parent_result.value + 1)


class ErrorScorer(Scorer):
    def __call__(self, outputs) -> Feedback:
        raise ValueError("Test error")


class ParentWithErrorChild(Scorer):
    _child_scorer: Scorer = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._child_scorer = ErrorScorer(name="error_scorer")

    def __call__(self, outputs) -> Feedback:
        try:
            self._child_scorer(outputs=outputs)
        except ValueError:
            pass  # Handle error
        return Feedback(name=self.name, value=10)


def get_scorer_call_events(mock_requests):
    """Extract ScorerCallEvent records from captured telemetry."""
    return [
        record for record in mock_requests if record["data"]["event_name"] == ScorerCallEvent.name
    ]


def get_event_params(mock_requests):
    """Get parsed params from ScorerCallEvent records."""
    scorer_call_events = get_scorer_call_events(mock_requests)
    return [json.loads(event["data"]["params"]) for event in scorer_call_events]


def test_nested_scorer_skips_telemetry(mock_requests, mock_telemetry_client: TelemetryClient):
    child = child_scorer_func
    parent = ParentScorer(name="parent_scorer", child_scorer=child)

    result = parent(outputs="test output")
    # Expected: len("test output") * 2 = 22
    assert result.value == 22

    mock_telemetry_client.flush()

    scorer_events = get_scorer_call_events(mock_requests)
    assert len(scorer_events) == 1

    event_params = get_event_params(mock_requests)
    assert len(event_params) == 1
    assert event_params[0]["scorer_class"] == "UserDefinedScorer"
    # This verifies it's the parent (kind="class"), not child (kind="decorator")
    assert event_params[0]["scorer_kind"] == "class"


def test_multi_level_nesting_skips_telemetry(mock_requests, mock_telemetry_client: TelemetryClient):
    child = child_scorer_func
    parent = parent_scorer_func(child_scorer=child)
    grandparent = GrandparentScorer(name="grandparent_scorer", parent_scorer=parent)

    result = grandparent(outputs="test")
    # Expected: (len("test") * 2) + 1 = 9
    assert result.value == 9

    mock_telemetry_client.flush()

    scorer_events = get_scorer_call_events(mock_requests)
    assert len(scorer_events) == 1

    event_params = get_event_params(mock_requests)
    assert len(event_params) == 1
    assert event_params[0]["scorer_class"] == "UserDefinedScorer"
    # Verifies it's the grandparent (kind="class"), not nested (kind="decorator")
    assert event_params[0]["scorer_kind"] == "class"


def test_recursive_scorer_skips_nested_telemetry(
    mock_requests, mock_telemetry_client: TelemetryClient
):
    recursive_scorer = RecursiveScorer(name="recursive_scorer", max_depth=5)

    result = recursive_scorer(outputs="test", depth=0)
    # Expected: len("test") = 4 (after recursing to max_depth=5)
    assert result.value == 4

    mock_telemetry_client.flush()

    scorer_events = get_scorer_call_events(mock_requests)
    assert len(scorer_events) == 1

    event_params = get_event_params(mock_requests)
    assert len(event_params) == 1


def test_thread_safety_concurrent_scorers(mock_requests, mock_telemetry_client: TelemetryClient):
    child = child_scorer_func

    results = []
    errors = []

    def run_scorer(scorer, outputs):
        try:
            result = scorer(outputs=outputs)
            results.append(result)
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(10):
        parent = ParentScorer(name=f"parent{i}", child_scorer=child)
        thread = threading.Thread(target=run_scorer, args=(parent, f"test{i}"))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert len(errors) == 0
    assert len(results) == 10

    mock_telemetry_client.flush()

    scorer_events = get_scorer_call_events(mock_requests)
    assert len(scorer_events) == 10

    event_params = get_event_params(mock_requests)
    assert len(event_params) == 10


def test_error_in_nested_scorer_still_records_parent_telemetry(
    mock_requests, mock_telemetry_client: TelemetryClient
):
    parent = ParentWithErrorChild(name="parent_scorer")

    result = parent(outputs="test")
    # Expected: 10 (hardcoded value returned after handling error)
    assert result.value == 10

    mock_telemetry_client.flush()

    scorer_events = get_scorer_call_events(mock_requests)
    assert len(scorer_events) == 1

    event_params = get_event_params(mock_requests)
    assert len(event_params) == 1


def test_direct_child_call_records_telemetry(mock_requests, mock_telemetry_client: TelemetryClient):
    child = child_scorer_func

    result = child(outputs="test")
    # Expected: len("test") = 4
    assert result == 4  # Decorator scorer returns int directly

    mock_telemetry_client.flush()

    scorer_events = get_scorer_call_events(mock_requests)
    assert len(scorer_events) == 1

    event_params = get_event_params(mock_requests)
    assert len(event_params) == 1
    # Verify it's the decorator scorer
    assert event_params[0]["scorer_kind"] == "decorator"


def test_sequential_parent_calls_each_record_telemetry(
    mock_requests, mock_telemetry_client: TelemetryClient
):
    child = child_scorer_func
    parent = ParentScorer(name="parent_scorer", child_scorer=child)

    result1 = parent(outputs="test1")
    result2 = parent(outputs="test2")
    # Expected: len("test1") * 2 = 10
    assert result1.value == 10
    # Expected: len("test2") * 2 = 10
    assert result2.value == 10

    mock_telemetry_client.flush()

    scorer_events = get_scorer_call_events(mock_requests)
    assert len(scorer_events) == 2

    event_params = get_event_params(mock_requests)
    assert len(event_params) == 2


def test_async_scorer_raises_error():
    with pytest.raises(TypeError, match="Async scorer '__call__' methods are not supported"):

        class AsyncScorer(Scorer):
            async def __call__(self, outputs) -> Feedback:
                await asyncio.sleep(0.001)
                return Feedback(name=self.name, value=len(outputs))


def test_telemetry_disabled_nested_scorers_work(
    mock_requests, mock_telemetry_client: TelemetryClient
):
    with mock.patch("mlflow.telemetry.track.is_telemetry_disabled", return_value=True):
        child = child_scorer_func
        parent = ParentScorer(name="parent_scorer", child_scorer=child)

        result = parent(outputs="test")
        # Expected: len("test") * 2 = 8
        assert result.value == 8

        mock_telemetry_client.flush()

        scorer_events = get_scorer_call_events(mock_requests)
        assert len(scorer_events) == 0


def test_decorator_scorer_with_nested_call(mock_requests, mock_telemetry_client: TelemetryClient):
    @scorer
    def nested_checker(outputs) -> int:
        return len(outputs)

    @scorer
    def parent_checker(outputs) -> int:
        nested_result = nested_checker(outputs=outputs)
        return nested_result * 2

    result = parent_checker(outputs="test")
    assert result == 8

    mock_telemetry_client.flush()

    scorer_events = get_scorer_call_events(mock_requests)
    assert len(scorer_events) == 1

    event_params = get_event_params(mock_requests)
    assert len(event_params) == 1
    assert event_params[0]["scorer_kind"] == "decorator"
