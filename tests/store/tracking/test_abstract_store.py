from abc import ABC
from unittest import mock
from unittest.mock import patch

import pytest

from mlflow.entities import Metric
from mlflow.entities.metric import MetricWithRunId
from mlflow.store.tracking.abstract_store import AbstractStore


class MockAbstractStore(AbstractStore, ABC):
    """Mock implementation of AbstractStore for testing."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def get_metric_history(self, run_id, metric_key, max_results=None, page_token=None):
        return [m for m in self.metrics if m.run_id == run_id and m.key == metric_key]


@pytest.fixture
def store():
    with patch.multiple(MockAbstractStore, __abstractmethods__=set()):
        yield MockAbstractStore()


@pytest.fixture
def mock_tracking_store():
    with mock.patch("mlflow.tracking._get_store") as mock_get_store:
        mock_store = mock.Mock()
        mock_get_store.return_value = mock_store
        yield mock_store


def test_get_metric_history_bulk_interval_empty_run_ids(store):
    result = store.get_metric_history_bulk_interval([], "accuracy", 10, 0, 100)
    assert result == []


def test_get_metric_history_bulk_interval_single_run_no_metrics(store):
    store.metrics = []
    result = store.get_metric_history_bulk_interval(["run1"], "accuracy", 10, 0, 100)
    assert result == []


def test_get_metric_history_bulk_interval_single_run_single_metric(store):
    store.metrics = [Metric("accuracy", 0.8, 1000, 5, run_id="run1")]

    result = store.get_metric_history_bulk_interval(["run1"], "accuracy", 10, 0, 100)
    assert len(result) == 1
    assert result[0].run_id == "run1"
    assert result[0].key == "accuracy"
    assert result[0].value == 0.8
    assert result[0].step == 5


def test_get_metric_history_bulk_interval_single_run_multiple_metrics_within_range(store):
    store.metrics = [
        Metric("accuracy", 0.7, 1000, 1, run_id="run1"),
        Metric("accuracy", 0.8, 2000, 5, run_id="run1"),
        Metric("accuracy", 0.9, 3000, 10, run_id="run1"),
    ]

    result = store.get_metric_history_bulk_interval(["run1"], "accuracy", 10, 0, 15)
    assert len(result) == 3
    assert [r.step for r in result] == [1, 5, 10]


def test_get_metric_history_bulk_interval_multiple_runs_same_steps(store):
    store.metrics = [
        Metric("accuracy", 0.7, 1000, 5, run_id="run1"),
        Metric("accuracy", 0.8, 1000, 5, run_id="run2"),
        Metric("accuracy", 0.75, 2000, 10, run_id="run1"),
        Metric("accuracy", 0.85, 2000, 10, run_id="run2"),
    ]

    result = store.get_metric_history_bulk_interval(["run1", "run2"], "accuracy", 10, 0, 20)
    assert len(result) == 4
    run1_results = [r for r in result if r.run_id == "run1"]
    run2_results = [r for r in result if r.run_id == "run2"]
    assert len(run1_results) == 2
    assert len(run2_results) == 2


def test_get_metric_history_bulk_interval_sampling_when_exceeds_max_results(store):
    store.metrics = []
    for step in range(0, 100, 2):  # 50 steps: 0, 2, 4, ..., 98
        store.metrics.append(
            Metric("accuracy", 0.5 + step / 200, 1000 + step * 10, step, run_id="run1")
        )

    result = store.get_metric_history_bulk_interval(["run1"], "accuracy", 10, 0, 98)
    assert len(result) <= 12  # max_results + some buffer for min/max

    steps = [r.step for r in result]
    assert 0 in steps  # min step should always be included
    assert len(steps) >= 10  # Should have at least max_results steps
    assert max(steps) >= 80  # Should include steps near the end of range


def test_get_metric_history_bulk_interval_none_start_end_steps(store):
    store.metrics = [
        Metric("accuracy", 0.7, 1000, 5, run_id="run1"),
        Metric("accuracy", 0.8, 2000, 15, run_id="run1"),
        Metric("accuracy", 0.9, 3000, 25, run_id="run1"),
    ]

    result = store.get_metric_history_bulk_interval(["run1"], "accuracy", 10, None, None)
    assert len(result) == 3


def test_get_metric_history_bulk_interval_different_metric_keys(store):
    store.metrics = [
        Metric("accuracy", 0.8, 1000, 5, run_id="run1"),
        Metric("loss", 0.2, 1000, 5, run_id="run1"),
        Metric("accuracy", 0.9, 2000, 10, run_id="run1"),
    ]

    result = store.get_metric_history_bulk_interval(["run1"], "accuracy", 10, 0, 20)
    assert len(result) == 2
    assert all(r.key == "accuracy" for r in result)


def test_get_metric_history_bulk_interval_metric_sorting_by_step_and_timestamp(store):
    store.metrics = [
        Metric("accuracy", 0.8, 3000, 5, run_id="run1"),  # Same step, later timestamp
        Metric("accuracy", 0.7, 1000, 5, run_id="run1"),  # Same step, earlier timestamp
        Metric("accuracy", 0.9, 2000, 10, run_id="run1"),  # Different step
    ]

    result = store.get_metric_history_bulk_interval(["run1"], "accuracy", 10, 0, 20)
    assert len(result) == 3
    assert result[0].timestamp == 1000  # step 5, earlier timestamp
    assert result[1].timestamp == 3000  # step 5, later timestamp
    assert result[2].step == 10


def test_get_metric_history_bulk_interval_bisect_boundary_conditions(store):
    store.metrics = [
        Metric("accuracy", 0.7, 1000, 10, run_id="run1"),  # Exact start boundary
        Metric("accuracy", 0.8, 2000, 15, run_id="run1"),
        Metric("accuracy", 0.9, 3000, 20, run_id="run1"),  # Exact end boundary
    ]

    result = store.get_metric_history_bulk_interval(["run1"], "accuracy", 10, 10, 20)
    assert len(result) == 3
    steps = [r.step for r in result]
    assert steps == [10, 15, 20]


@pytest.mark.parametrize(
    ("start_step", "end_step", "expected_count"),
    [
        (10, 20, 0),  # Metrics outside range
        (10, 5, 0),  # Invalid range (start > end)
        (10, 20, 1),  # Single step in range
    ],
)
def test_get_metric_history_bulk_interval_edge_cases(store, start_step, end_step, expected_count):
    if start_step == 10 and end_step == 20 and expected_count == 1:
        # Single step in range case
        store.metrics = [
            Metric("accuracy", 0.7, 1000, 1, run_id="run1"),
            Metric("accuracy", 0.8, 2000, 15, run_id="run1"),  # Only this one in range
            Metric("accuracy", 0.9, 3000, 25, run_id="run1"),
        ]
    else:
        # Outside range or invalid range cases
        store.metrics = [
            Metric("accuracy", 0.7, 1000, 1, run_id="run1"),
            Metric("accuracy", 0.8, 2000, 50, run_id="run1"),
            Metric("accuracy", 0.9, 3000, 100, run_id="run1"),
        ]

    result = store.get_metric_history_bulk_interval(["run1"], "accuracy", 10, start_step, end_step)
    assert len(result) == expected_count
    if expected_count == 1:
        assert result[0].step == 15


@pytest.mark.parametrize(
    (
        "start_step",
        "end_step",
        "max_results",
        "steps",
        "should_include_min",
        "should_include_max_or_near",
    ),
    [
        # Evenly spaced sampling - should include min, may not include exact max
        (0, 10, 5, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], True, True),
        # Clipped list shorter than max_results returns everything
        (4, 8, 5, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], True, True),
        # Works with interval-logged steps - should include min
        (0, 100, 5, [0, 20, 40, 60, 80, 100], True, True),
        (0, 1000, 5, list(range(0, 1001, 10)), True, True),
    ],
)
def test_get_metric_history_bulk_interval_sampling_algorithm(
    store, start_step, end_step, max_results, steps, should_include_min, should_include_max_or_near
):
    store.metrics = [Metric("accuracy", 0.5, 1000 + s, s, run_id="run1") for s in steps]

    result = store.get_metric_history_bulk_interval(
        ["run1"], "accuracy", max_results, start_step, end_step
    )
    result_steps = {r.step for r in result}

    # Test key properties rather than exact step sets
    if should_include_min:
        assert start_step in result_steps or min(result_steps) >= start_step

    if should_include_max_or_near:
        # Should include steps near the end of the range
        max_result_step = max(result_steps)
        assert max_result_step >= end_step * 0.8  # At least 80% of the way to end_step

    # Should not exceed max_results by much (allowing for min/max inclusion)
    assert len(result_steps) <= max_results + 2


@pytest.mark.parametrize(
    ("start_step", "end_step", "max_results", "steps", "expected"),
    [
        # should be evenly spaced and include the beginning and
        # end despite sometimes making it go above max_results
        (0, 10, 5, list(range(10)), {0, 2, 4, 6, 8, 9}),
        # if the clipped list is shorter than max_results,
        # then everything will be returned
        (4, 8, 5, list(range(10)), {4, 5, 6, 7, 8}),
        # works if steps are logged in intervals
        (0, 100, 5, list(range(0, 101, 20)), {0, 20, 40, 60, 80, 100}),
        (0, 1000, 5, list(range(0, 1001, 10)), {0, 200, 400, 600, 800, 1000}),
        (1000, 1100, 50, list(range(900, 1200)), set(range(1000, 1100 + 1, 2))),
    ],
)
def test_get_sampled_steps_from_steps(start_step, end_step, max_results, steps, expected, store):
    run_id = "run1"
    metric_key = "accuracy"
    store.metrics = [Metric(metric_key, 0.0, 1000, step, run_id=run_id) for step in steps]
    metrics = store.get_metric_history_bulk_interval(
        [run_id], metric_key, max_results, start_step, end_step
    )

    actual_steps = {metric.step for metric in metrics}
    assert actual_steps == expected


# Tests for get_metric_history_bulk_interval_from_steps


def test_get_metric_history_bulk_interval_from_steps_empty_steps(store):
    store.metrics = [Metric("accuracy", 0.8, 1000, 5, run_id="run1")]
    result = store.get_metric_history_bulk_interval_from_steps("run1", "accuracy", [], 10)
    assert result == []


def test_get_metric_history_bulk_interval_from_steps_no_matching_steps(store):
    store.metrics = [
        Metric("accuracy", 0.8, 1000, 5, run_id="run1"),
        Metric("accuracy", 0.9, 2000, 10, run_id="run1"),
    ]
    result = store.get_metric_history_bulk_interval_from_steps("run1", "accuracy", [1, 2, 3], 10)
    assert result == []


def test_get_metric_history_bulk_interval_from_steps_single_matching_step(store):
    store.metrics = [
        Metric("accuracy", 0.8, 1000, 5, run_id="run1"),
        Metric("accuracy", 0.9, 2000, 10, run_id="run1"),
    ]
    result = store.get_metric_history_bulk_interval_from_steps("run1", "accuracy", [5], 10)

    assert len(result) == 1
    assert isinstance(result[0], MetricWithRunId)
    assert result[0].run_id == "run1"
    assert result[0].key == "accuracy"
    assert result[0].value == 0.8
    assert result[0].step == 5


def test_get_metric_history_bulk_interval_from_steps_multiple_matching_steps(store):
    store.metrics = [
        Metric("accuracy", 0.8, 1000, 5, run_id="run1"),
        Metric("accuracy", 0.9, 2000, 10, run_id="run1"),
        Metric("accuracy", 0.85, 1500, 7, run_id="run1"),
    ]
    result = store.get_metric_history_bulk_interval_from_steps("run1", "accuracy", [5, 10], 10)

    assert len(result) == 2
    # Should be sorted by step, then timestamp
    assert result[0].step == 5
    assert result[1].step == 10


def test_get_metric_history_bulk_interval_from_steps_max_results_limit(store):
    store.metrics = [
        Metric("accuracy", 0.8, 1000, 5, run_id="run1"),
        Metric("accuracy", 0.9, 2000, 10, run_id="run1"),
        Metric("accuracy", 0.85, 1500, 15, run_id="run1"),
    ]
    result = store.get_metric_history_bulk_interval_from_steps("run1", "accuracy", [5, 10, 15], 2)

    assert len(result) == 2
    # Should return first 2 after sorting
    assert result[0].step == 5
    assert result[1].step == 10


def test_get_metric_history_bulk_interval_from_steps_sorting_by_step_and_timestamp(store):
    store.metrics = [
        Metric("accuracy", 0.8, 2000, 5, run_id="run1"),  # Later timestamp
        Metric("accuracy", 0.7, 1000, 5, run_id="run1"),  # Earlier timestamp, same step
        Metric("accuracy", 0.9, 1500, 10, run_id="run1"),
    ]
    result = store.get_metric_history_bulk_interval_from_steps("run1", "accuracy", [5, 10], 10)

    assert len(result) == 3
    # Step 5 metrics should come first, sorted by timestamp
    assert result[0].step == 5
    assert result[0].timestamp == 1000
    assert result[1].step == 5
    assert result[1].timestamp == 2000
    assert result[2].step == 10


def test_get_metric_history_bulk_interval_from_steps_different_run_id(store):
    store.metrics = [
        Metric("accuracy", 0.8, 1000, 5, run_id="run1"),
        Metric("accuracy", 0.9, 2000, 5, run_id="run2"),
    ]
    result = store.get_metric_history_bulk_interval_from_steps("run1", "accuracy", [5], 10)

    assert len(result) == 1
    assert result[0].run_id == "run1"
    assert result[0].value == 0.8


def test_get_metric_history_bulk_interval_from_steps_different_metric_key(store):
    store.metrics = [
        Metric("accuracy", 0.8, 1000, 5, run_id="run1"),
        Metric("loss", 0.2, 1000, 5, run_id="run1"),
    ]
    result = store.get_metric_history_bulk_interval_from_steps("run1", "accuracy", [5], 10)

    assert len(result) == 1
    assert result[0].key == "accuracy"
