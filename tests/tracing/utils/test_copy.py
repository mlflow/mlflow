import time

import pytest

import mlflow
from mlflow.tracing.utils.copy import copy_trace_to_experiment

from tests.tracing.helper import purge_traces


def _create_test_span_dict(request_id="test-trace", parent_id=None):
    """Helper to create a minimal valid span dict for testing"""
    return {
        "name": "root_span" if parent_id is None else "child_span",
        "context": {
            "span_id": "0d48a6670588966b" if parent_id is None else "6fc32f36ef591f60",
            "trace_id": "63076d0c1b90f1df0970f897dc428bd6",
        },
        "parent_id": parent_id,
        "start_time": 100,
        "end_time": 200,
        "status_code": "OK",
        "status_message": "",
        "attributes": {
            "mlflow.traceRequestId": f'"{request_id}"',
            "mlflow.spanType": '"UNKNOWN"',
        },
        "events": [],
    }


@pytest.fixture(autouse=True)
def setup_experiment():
    """Set up a test experiment before each test"""
    exp = mlflow.set_experiment(f"test_copy_trace_{time.time()}")
    yield exp
    purge_traces(exp.experiment_id)


def test_copy_trace_with_metadata():
    trace_dict = {
        "info": {
            "request_id": "test-trace-789",
            "experiment_id": "0",
            "timestamp_ms": 100,
            "execution_time_ms": 200,
            "status": "OK",
            "trace_metadata": {
                "mlflow.trace.session": "session123",
                "custom.metadata": "metadata_value",
                "user.key": "user_value",
            },
        },
        "data": {"spans": [_create_test_span_dict("test-trace-789")]},
    }

    new_trace_id = copy_trace_to_experiment(trace_dict)

    # Verify metadata was copied correctly
    trace = mlflow.get_trace(new_trace_id)
    metadata = trace.info.trace_metadata

    assert metadata["mlflow.trace.session"] == "session123"
    assert metadata["custom.metadata"] == "metadata_value"
    assert metadata["user.key"] == "user_value"


def test_copy_trace_missing_info():
    trace_dict = {"data": {"spans": [_create_test_span_dict("test-trace-no-info")]}}

    # Should not raise an error, just skip tag/metadata copying
    new_trace_id = copy_trace_to_experiment(trace_dict)

    assert new_trace_id is not None
    trace = mlflow.get_trace(new_trace_id)
    assert trace is not None


def test_copy_trace_missing_metadata():
    trace_dict = {
        "info": {
            "request_id": "test-trace-no-metadata",
            "experiment_id": "0",
            "tags": {
                "user.tag": "tag_value",
            },
        },
        "data": {"spans": [_create_test_span_dict("test-trace-no-metadata")]},
    }

    # Should not raise an error, just skip metadata copying
    new_trace_id = copy_trace_to_experiment(trace_dict)

    assert new_trace_id is not None
    trace = mlflow.get_trace(new_trace_id)

    # Tags should still be copied
    tags = trace.info.tags
    assert tags["user.tag"] == "tag_value"


def test_copy_trace_empty_metadata_dict():
    trace_dict = {
        "info": {
            "request_id": "test-trace-empty-metadata",
            "experiment_id": "0",
            "tags": {
                "user.tag": "value",
            },
            "trace_metadata": {},
        },
        "data": {"spans": [_create_test_span_dict("test-trace-empty-metadata")]},
    }

    # Should not raise an error
    new_trace_id = copy_trace_to_experiment(trace_dict)

    assert new_trace_id is not None
    trace = mlflow.get_trace(new_trace_id)

    # Tags should still be copied
    tags = trace.info.tags
    assert tags["user.tag"] == "value"
