"""Tests for the OSS-native review-assignment fluent SDK.

These cover the fluent functions in ``mlflow.genai.review_assignments``,
which delegate to the tracking server's REST surface via ``TracingClient``.
"""

from unittest.mock import patch

from mlflow.genai.review_assignments import (
    BulkCreateFailure,
    BulkCreateReviewAssignmentsResult,
    ReviewAssignment,
    ReviewAssignmentState,
    ReviewTargetType,
    bulk_create_review_assignments,
    create_review_assignment,
    delete_review_assignment,
    get_review_assignment,
    list_review_assignments,
    list_review_assignments_for_target,
    update_review_assignment,
)
from mlflow.store.entities.paged_list import PagedList

_BASE = "mlflow.genai.review_assignments.TracingClient"


def _assignment(assignment_id="ra-1", state=ReviewAssignmentState.PENDING):
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
    )


def test_create_review_assignment_delegates():
    with patch(f"{_BASE}._create_review_assignment", return_value=_assignment()) as mock_create:
        result = create_review_assignment(
            "1", target_id="tr-1", reviewer="sme@example.com", assigner="kris@example.com"
        )
        assert result.assignment_id == "ra-1"
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        assert args == ("1",)
        assert kwargs["target_type"] == ReviewTargetType.TRACE
        assert kwargs["target_id"] == "tr-1"
        assert kwargs["reviewer"] == "sme@example.com"
        assert kwargs["assigner"] == "kris@example.com"


def test_bulk_create_review_assignments_delegates():
    result_obj = BulkCreateReviewAssignmentsResult(
        created=[_assignment("ra-1")],
        existing=["ra-2"],
        failed=[BulkCreateFailure("tr-2", "b@example.com", "boom")],
    )
    with patch(f"{_BASE}._bulk_create_review_assignments", return_value=result_obj) as mock_bulk:
        result = bulk_create_review_assignments(
            "1",
            target_ids=["tr-1", "tr-2"],
            reviewers=["a@example.com", "b@example.com"],
            assigner="kris@example.com",
        )
        assert result.existing == ["ra-2"]
        kwargs = mock_bulk.call_args[1]
        assert kwargs["target_ids"] == ["tr-1", "tr-2"]
        assert kwargs["reviewers"] == ["a@example.com", "b@example.com"]
        assert kwargs["target_type"] == ReviewTargetType.TRACE


def test_get_review_assignment_delegates():
    with patch(f"{_BASE}._get_review_assignment", return_value=_assignment()) as mock_get:
        result = get_review_assignment("ra-1")
        assert result.assignment_id == "ra-1"
        mock_get.assert_called_once_with("ra-1")


def test_list_review_assignments_delegates():
    paged = PagedList([_assignment()], "tok")
    with patch(f"{_BASE}._list_review_assignments", return_value=paged) as mock_list:
        result = list_review_assignments(
            experiment_id="1", reviewer="sme@example.com", state="in_progress", max_results=25
        )
        assert result.token == "tok"
        kwargs = mock_list.call_args[1]
        assert kwargs["experiment_id"] == "1"
        assert kwargs["reviewer"] == "sme@example.com"
        assert kwargs["state"] == "in_progress"
        assert kwargs["max_results"] == 25


def test_list_review_assignments_for_target_delegates():
    with patch(
        f"{_BASE}._list_review_assignments_for_target", return_value=[_assignment()]
    ) as mock_list:
        result = list_review_assignments_for_target("tr-1")
        assert len(result) == 1
        mock_list.assert_called_once_with("tr-1")


def test_update_review_assignment_delegates():
    with patch(
        f"{_BASE}._update_review_assignment",
        return_value=_assignment(state=ReviewAssignmentState.COMPLETE),
    ) as mock_update:
        result = update_review_assignment("ra-1", state="complete")
        assert result.state == ReviewAssignmentState.COMPLETE
        mock_update.assert_called_once_with("ra-1", state="complete")


def test_delete_review_assignment_delegates():
    with patch(f"{_BASE}._delete_review_assignment", return_value=None) as mock_delete:
        delete_review_assignment("ra-1")
        mock_delete.assert_called_once_with("ra-1")
