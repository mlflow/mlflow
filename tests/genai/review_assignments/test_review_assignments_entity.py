import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.review_assignments.review_assignments import (
    BulkCreateFailure,
    ReviewAssignment,
    ReviewAssignmentState,
    ReviewTargetType,
)
from mlflow.protos.review_assignments_pb2 import (
    COMPLETE,
    PENDING,
    REVIEW_ASSIGNMENT_STATE_UNSPECIFIED,
    REVIEW_TARGET_TYPE_UNSPECIFIED,
    TRACE,
)


@pytest.mark.parametrize("completed_time_ms", [None, 300])
def test_review_assignment_proto_roundtrip(completed_time_ms):
    assignment = ReviewAssignment(
        assignment_id="ra-1",
        experiment_id="42",
        target_type=ReviewTargetType.TRACE,
        target_id="tr-abc",
        reviewer="sme@example.com",
        assigner="kris@example.com",
        state=ReviewAssignmentState.COMPLETE,
        creation_time_ms=100,
        last_update_time_ms=200,
        completed_time_ms=completed_time_ms,
    )
    assert ReviewAssignment.from_proto(assignment.to_proto()) == assignment


def test_completed_time_ms_absent_when_none():
    assignment = ReviewAssignment(
        assignment_id="ra-1",
        experiment_id="42",
        target_type=ReviewTargetType.TRACE,
        target_id="tr-abc",
        reviewer="sme@example.com",
        assigner="kris@example.com",
        state=ReviewAssignmentState.PENDING,
        creation_time_ms=100,
        last_update_time_ms=200,
    )
    assert assignment.to_proto().HasField("completed_time_ms") is False


@pytest.mark.parametrize(
    ("state", "proto_value"),
    [
        (ReviewAssignmentState.PENDING, PENDING),
        (ReviewAssignmentState.COMPLETE, COMPLETE),
    ],
)
def test_state_to_from_proto(state, proto_value):
    assert state.to_proto() == proto_value
    assert ReviewAssignmentState.from_proto(proto_value) is state


def test_target_type_to_from_proto():
    assert ReviewTargetType.TRACE.to_proto() == TRACE
    assert ReviewTargetType.from_proto(TRACE) is ReviewTargetType.TRACE


def test_target_type_from_proto_rejects_unspecified():
    with pytest.raises(MlflowException, match="`target_type` must be one of"):
        ReviewTargetType.from_proto(REVIEW_TARGET_TYPE_UNSPECIFIED)


def test_state_from_proto_rejects_unspecified():
    with pytest.raises(MlflowException, match="`state` must be one of"):
        ReviewAssignmentState.from_proto(REVIEW_ASSIGNMENT_STATE_UNSPECIFIED)


def test_bulk_create_failure_to_proto():
    proto = BulkCreateFailure("tr-1", "sme@example.com", "boom").to_proto()
    assert proto.target_id == "tr-1"
    assert proto.reviewer == "sme@example.com"
    assert proto.error_message == "boom"
