"""Unit tests for the OSS-native review-assignment REST handlers.

These exercise the handler functions directly (mocking the tracking store
and request-message parsing); the wire-format round-trip through
``ReviewAssignment.to_proto`` / ``from_proto`` is covered separately.
"""

import json
from unittest import mock

from mlflow.genai.review_assignments.review_assignments import (
    BulkCreateFailure,
    BulkCreateReviewAssignmentsResult,
    ReviewAssignment,
    ReviewAssignmentState,
    ReviewTargetType,
)
from mlflow.protos.review_assignments_pb2 import (
    COMPLETE,
    IN_PROGRESS,
    PENDING,
    REVIEW_ASSIGNMENT_STATE_UNSPECIFIED,
    REVIEW_TARGET_TYPE_UNSPECIFIED,
    TRACE,
    BulkCreateReviewAssignments,
    CreateReviewAssignment,
    DeleteReviewAssignment,
    GetReviewAssignment,
    ListReviewAssignments,
    ListReviewAssignmentsForTarget,
    UpdateReviewAssignment,
)
from mlflow.server.handlers import (
    _bulk_create_review_assignments,
    _create_review_assignment,
    _delete_review_assignment,
    _get_review_assignment,
    _list_review_assignments,
    _list_review_assignments_for_target,
    _update_review_assignment,
)
from mlflow.store.entities.paged_list import PagedList

_BASE_PATCH = "mlflow.server.handlers"


def _assignment(
    assignment_id: str = "ra-1",
    state: ReviewAssignmentState = ReviewAssignmentState.PENDING,
    completed_time_ms: int | None = None,
) -> ReviewAssignment:
    return ReviewAssignment(
        assignment_id=assignment_id,
        experiment_id="1",
        target_type=ReviewTargetType.TRACE,
        target_id="tr-1",
        reviewer="sme@example.com",
        assigner="kris@example.com",
        state=state,
        creation_time_ms=100,
        last_update_time_ms=200,
        completed_time_ms=completed_time_ms,
    )


def _run_handler(handler, request_message, store_attr: str, return_value):
    with (
        mock.patch(f"{_BASE_PATCH}._get_tracking_store") as mock_store,
        mock.patch(f"{_BASE_PATCH}._get_request_message", return_value=request_message),
    ):
        getattr(mock_store.return_value, store_attr).return_value = return_value
        response = handler()
        return mock_store.return_value, response


def test_create_review_assignment_routes_args():
    request_message = CreateReviewAssignment(
        experiment_id="1",
        target_type=TRACE,
        target_id="tr-1",
        reviewer="sme@example.com",
        assigner="kris@example.com",
    )
    store, response = _run_handler(
        _create_review_assignment, request_message, "create_review_assignment", _assignment()
    )
    store.create_review_assignment.assert_called_once()
    call_kwargs = store.create_review_assignment.call_args[1]
    assert call_kwargs["experiment_id"] == "1"
    assert call_kwargs["target_type"] == ReviewTargetType.TRACE
    assert call_kwargs["target_id"] == "tr-1"
    assert call_kwargs["reviewer"] == "sme@example.com"
    assert call_kwargs["assigner"] == "kris@example.com"

    body = json.loads(response.get_data())
    assert body["review_assignment"]["assignment_id"] == "ra-1"
    assert body["review_assignment"]["state"] == "PENDING"


def test_create_review_assignment_rejects_unspecified_target_type():
    request_message = CreateReviewAssignment(
        experiment_id="1",
        target_type=REVIEW_TARGET_TYPE_UNSPECIFIED,
        target_id="tr-1",
        reviewer="sme@example.com",
        assigner="kris@example.com",
    )
    with (
        mock.patch(f"{_BASE_PATCH}._get_tracking_store") as mock_store,
        mock.patch(f"{_BASE_PATCH}._get_request_message", return_value=request_message),
    ):
        response = _create_review_assignment()
        assert response.status_code == 400
        assert json.loads(response.get_data())["error_code"] == "INVALID_PARAMETER_VALUE"
        mock_store.return_value.create_review_assignment.assert_not_called()


def test_bulk_create_review_assignments_routes_buckets():
    request_message = BulkCreateReviewAssignments(
        experiment_id="1",
        target_type=TRACE,
        target_ids=["tr-1", "tr-2"],
        reviewers=["a@example.com", "b@example.com"],
        assigner="kris@example.com",
    )
    result = BulkCreateReviewAssignmentsResult(
        created=[_assignment("ra-1"), _assignment("ra-2")],
        existing=["ra-3"],
        failed=[BulkCreateFailure("tr-2", "b@example.com", "boom")],
    )
    store, response = _run_handler(
        _bulk_create_review_assignments,
        request_message,
        "bulk_create_review_assignments",
        result,
    )
    store.bulk_create_review_assignments.assert_called_once()
    call_kwargs = store.bulk_create_review_assignments.call_args[1]
    assert call_kwargs["experiment_id"] == "1"
    assert call_kwargs["target_type"] == ReviewTargetType.TRACE
    assert call_kwargs["target_ids"] == ["tr-1", "tr-2"]
    assert call_kwargs["reviewers"] == ["a@example.com", "b@example.com"]
    assert call_kwargs["assigner"] == "kris@example.com"

    body = json.loads(response.get_data())
    assert [c["assignment_id"] for c in body["created"]] == ["ra-1", "ra-2"]
    assert body["existing"] == ["ra-3"]
    assert body["failed"][0]["error_message"] == "boom"


def test_bulk_create_review_assignments_rejects_unspecified_target_type():
    request_message = BulkCreateReviewAssignments(
        experiment_id="1",
        target_type=REVIEW_TARGET_TYPE_UNSPECIFIED,
        target_ids=["tr-1"],
        reviewers=["sme@example.com"],
        assigner="kris@example.com",
    )
    with (
        mock.patch(f"{_BASE_PATCH}._get_tracking_store") as mock_store,
        mock.patch(f"{_BASE_PATCH}._get_request_message", return_value=request_message),
    ):
        response = _bulk_create_review_assignments()
        assert response.status_code == 400
        assert json.loads(response.get_data())["error_code"] == "INVALID_PARAMETER_VALUE"
        mock_store.return_value.bulk_create_review_assignments.assert_not_called()


def test_get_review_assignment_routes_args():
    request_message = GetReviewAssignment(assignment_id="ra-1")
    store, response = _run_handler(
        _get_review_assignment, request_message, "get_review_assignment", _assignment()
    )
    store.get_review_assignment.assert_called_once_with("ra-1")
    body = json.loads(response.get_data())
    assert body["review_assignment"]["assignment_id"] == "ra-1"


def test_list_review_assignments_routes_filters():
    request_message = ListReviewAssignments(
        experiment_id="1",
        reviewer="sme@example.com",
        state=IN_PROGRESS,
        target_type=TRACE,
        max_results=50,
    )
    paged = PagedList([_assignment(state=ReviewAssignmentState.IN_PROGRESS)], "next-tok")
    store, response = _run_handler(
        _list_review_assignments, request_message, "list_review_assignments", paged
    )
    call_kwargs = store.list_review_assignments.call_args[1]
    assert call_kwargs["experiment_id"] == "1"
    assert call_kwargs["reviewer"] == "sme@example.com"
    assert call_kwargs["state"] == ReviewAssignmentState.IN_PROGRESS
    assert call_kwargs["target_type"] == ReviewTargetType.TRACE
    assert call_kwargs["max_results"] == 50

    body = json.loads(response.get_data())
    assert body["next_page_token"] == "next-tok"
    assert len(body["review_assignments"]) == 1


def test_list_review_assignments_treats_unspecified_filters_as_absent():
    request_message = ListReviewAssignments(
        experiment_id="1",
        state=REVIEW_ASSIGNMENT_STATE_UNSPECIFIED,
        target_type=REVIEW_TARGET_TYPE_UNSPECIFIED,
    )
    paged = PagedList([_assignment()], None)
    store, _ = _run_handler(
        _list_review_assignments, request_message, "list_review_assignments", paged
    )
    call_kwargs = store.list_review_assignments.call_args[1]
    assert call_kwargs["state"] is None
    assert call_kwargs["target_type"] is None


def test_list_review_assignments_normalizes_empty_scope_to_none():
    # An explicit empty-string experiment_id/reviewer must not slip past
    # the store's `is None` scope guard (would crash on int("")).
    request_message = ListReviewAssignments(experiment_id="", reviewer="sme@example.com")
    paged = PagedList([_assignment()], None)
    store, _ = _run_handler(
        _list_review_assignments, request_message, "list_review_assignments", paged
    )
    call_kwargs = store.list_review_assignments.call_args[1]
    assert call_kwargs["experiment_id"] is None
    assert call_kwargs["reviewer"] == "sme@example.com"


def test_list_review_assignments_for_target_routes_args():
    request_message = ListReviewAssignmentsForTarget(target_id="tr-1")
    store, response = _run_handler(
        _list_review_assignments_for_target,
        request_message,
        "list_review_assignments_for_target",
        [_assignment("ra-1"), _assignment("ra-2")],
    )
    store.list_review_assignments_for_target.assert_called_once_with("tr-1")
    body = json.loads(response.get_data())
    assert len(body["review_assignments"]) == 2


def test_update_review_assignment_routes_state():
    request_message = UpdateReviewAssignment(assignment_id="ra-1", state=COMPLETE)
    store, response = _run_handler(
        _update_review_assignment,
        request_message,
        "update_review_assignment",
        _assignment(state=ReviewAssignmentState.COMPLETE, completed_time_ms=300),
    )
    store.update_review_assignment.assert_called_once_with(
        "ra-1", state=ReviewAssignmentState.COMPLETE
    )
    body = json.loads(response.get_data())
    assert body["review_assignment"]["state"] == "COMPLETE"
    assert body["review_assignment"]["completed_time_ms"] == 300


def test_update_review_assignment_rejects_unspecified_state():
    request_message = UpdateReviewAssignment(
        assignment_id="ra-1", state=REVIEW_ASSIGNMENT_STATE_UNSPECIFIED
    )
    with (
        mock.patch(f"{_BASE_PATCH}._get_tracking_store") as mock_store,
        mock.patch(f"{_BASE_PATCH}._get_request_message", return_value=request_message),
    ):
        response = _update_review_assignment()
        assert response.status_code == 400
        assert json.loads(response.get_data())["error_code"] == "INVALID_PARAMETER_VALUE"
        mock_store.return_value.update_review_assignment.assert_not_called()


def test_delete_review_assignment_routes_args():
    request_message = DeleteReviewAssignment(assignment_id="ra-1")
    store, response = _run_handler(
        _delete_review_assignment, request_message, "delete_review_assignment", None
    )
    store.delete_review_assignment.assert_called_once_with("ra-1")
    assert response.status_code == 200


def test_proto_state_roundtrip_constants_match():
    assert ReviewAssignmentState.PENDING.to_proto() == PENDING
    assert ReviewAssignmentState.IN_PROGRESS.to_proto() == IN_PROGRESS
    assert ReviewAssignmentState.COMPLETE.to_proto() == COMPLETE
    assert ReviewTargetType.TRACE.to_proto() == TRACE
