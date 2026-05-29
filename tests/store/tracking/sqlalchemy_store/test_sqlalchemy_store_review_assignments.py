import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.review_assignments import (
    ReviewAssignment,
    ReviewAssignmentState,
    ReviewTargetType,
)
from mlflow.store.tracking.dbmodels.models import SqlReviewAssignment

from tests.store.tracking.sqlalchemy_store.conftest import _create_experiments

pytestmark = pytest.mark.notrackingurimock


def _create_assignment(store, experiment_id, *, target_id="tr-1", reviewer="sme@example.com"):
    return store.create_review_assignment(
        experiment_id=experiment_id,
        target_type="trace",
        target_id=target_id,
        reviewer=reviewer,
        assigner="kris@example.com",
    )


# ---------------------------------------------------------------------------
# create_review_assignment
# ---------------------------------------------------------------------------


def test_create_review_assignment_happy_path(store):
    exp_id = _create_experiments(store, "test_create_ra_happy")

    assignment = _create_assignment(store, exp_id)

    assert assignment.assignment_id.startswith(SqlReviewAssignment.ASSIGNMENT_ID_PREFIX)
    assert assignment.experiment_id == exp_id
    assert assignment.target_type == ReviewTargetType.TRACE
    assert assignment.target_id == "tr-1"
    assert assignment.reviewer == "sme@example.com"
    assert assignment.assigner == "kris@example.com"
    assert assignment.state == ReviewAssignmentState.PENDING
    assert assignment.creation_time_ms > 0
    assert assignment.last_update_time_ms == assignment.creation_time_ms
    assert assignment.completed_time_ms is None


def test_create_review_assignment_is_idempotent_on_duplicate_pair(store):
    exp_id = _create_experiments(store, "test_create_ra_idempotent")

    first = _create_assignment(store, exp_id)
    second = _create_assignment(store, exp_id)

    assert first.assignment_id == second.assignment_id
    assert first.creation_time_ms == second.creation_time_ms


def test_create_review_assignment_normalizes_reviewer_casing(store):
    # Two creates with differing-case reviewers must collapse to the
    # same row (case-insensitive identity via lowercasing on write).
    exp_id = _create_experiments(store, "test_create_ra_case")
    first = _create_assignment(store, exp_id, reviewer="Alice@Example.com")
    second = _create_assignment(store, exp_id, reviewer="ALICE@example.COM")
    assert first.assignment_id == second.assignment_id
    assert first.reviewer == "alice@example.com"


def test_create_review_assignment_strips_whitespace(store):
    exp_id = _create_experiments(store, "test_create_ra_strip")
    first = _create_assignment(store, exp_id, target_id="tr-1", reviewer="sme@example.com")
    second = _create_assignment(store, exp_id, target_id="  tr-1  ", reviewer="\tsme@example.com\n")
    assert first.assignment_id == second.assignment_id


def test_create_review_assignment_rejects_unknown_experiment(store):
    with pytest.raises(MlflowException, match="(?i)experiment"):
        _create_assignment(store, "999999")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"target_id": ""}, "target_id", id="empty-target-id"),
        pytest.param({"target_id": "   "}, "target_id", id="whitespace-target-id"),
        pytest.param({"reviewer": ""}, "reviewer", id="empty-reviewer"),
        pytest.param({"reviewer": "a" * 251}, "at most 250", id="reviewer-too-long"),
    ],
)
def test_create_review_assignment_validation_errors(store, kwargs, match):
    exp_id = _create_experiments(store, f"test_create_ra_invalid_{kwargs}")
    base = {"target_id": "tr-1", "reviewer": "sme@example.com"}
    with pytest.raises(MlflowException, match=match):
        store.create_review_assignment(
            experiment_id=exp_id,
            target_type="trace",
            assigner="kris@example.com",
            **(base | kwargs),
        )


def test_create_review_assignment_rejects_invalid_target_type(store):
    exp_id = _create_experiments(store, "test_create_ra_bad_target_type")
    with pytest.raises(MlflowException, match="target_type"):
        store.create_review_assignment(
            experiment_id=exp_id,
            target_type="not_a_real_type",
            target_id="tr-1",
            reviewer="sme@example.com",
            assigner="kris@example.com",
        )


# ---------------------------------------------------------------------------
# bulk_create_review_assignments
# ---------------------------------------------------------------------------


def test_bulk_create_review_assignments_cross_product(store):
    exp_id = _create_experiments(store, "test_bulk_create_ra")

    result = store.bulk_create_review_assignments(
        experiment_id=exp_id,
        target_type="trace",
        target_ids=["tr-1", "tr-2", "tr-3"],
        reviewers=["sme1@example.com", "sme2@example.com"],
        assigner="kris@example.com",
    )

    # 3 targets x 2 reviewers = 6 new rows.
    assert len(result.created) == 6
    assert result.existing == []
    assert result.failed == []
    # Identity check: no two created rows share a (target_id, reviewer) pair.
    pairs = {(a.target_id, a.reviewer) for a in result.created}
    assert len(pairs) == 6


def test_bulk_create_review_assignments_partial_existing(store):
    exp_id = _create_experiments(store, "test_bulk_partial_existing")
    # Pre-seed one of the rows the bulk call would create.
    first = _create_assignment(store, exp_id, target_id="tr-1", reviewer="sme1@example.com")

    result = store.bulk_create_review_assignments(
        experiment_id=exp_id,
        target_type="trace",
        target_ids=["tr-1", "tr-2"],
        reviewers=["sme1@example.com", "sme2@example.com"],
        assigner="kris@example.com",
    )

    # 4 total pairs; 1 already existed.
    assert len(result.created) == 3
    assert result.existing == [first.assignment_id]
    assert result.failed == []


def test_bulk_create_review_assignments_validation_failures(store):
    exp_id = _create_experiments(store, "test_bulk_validation_failures")

    result = store.bulk_create_review_assignments(
        experiment_id=exp_id,
        target_type="trace",
        target_ids=["tr-1", ""],  # second target_id is invalid
        reviewers=["sme1@example.com", ""],  # second reviewer is invalid
        assigner="kris@example.com",
    )

    # Valid pairs land in created (1*1); invalid pairs land in failed
    # (the other 3 cross-product entries each fail at least one
    # validation rule).
    assert len(result.created) == 1
    assert result.created[0].target_id == "tr-1"
    assert result.created[0].reviewer == "sme1@example.com"
    assert len(result.failed) == 3


def test_bulk_create_review_assignments_empty_input(store):
    # Empty target list -> empty result; no DB write attempted.
    exp_id = _create_experiments(store, "test_bulk_empty")
    result = store.bulk_create_review_assignments(
        experiment_id=exp_id,
        target_type="trace",
        target_ids=[],
        reviewers=["sme@example.com"],
        assigner="kris@example.com",
    )
    assert result.created == []
    assert result.existing == []
    assert result.failed == []


def test_bulk_create_review_assignments_rejects_unknown_experiment_early(store):
    # Bad experiment_id raises BEFORE the per-pair pre-flight fills
    # `failed` with N*M misleading errors.
    with pytest.raises(MlflowException, match="(?i)experiment"):
        store.bulk_create_review_assignments(
            experiment_id="999999",
            target_type="trace",
            target_ids=["tr-1"],
            reviewers=["sme@example.com"],
            assigner="kris@example.com",
        )


# ---------------------------------------------------------------------------
# get / list
# ---------------------------------------------------------------------------


def test_get_review_assignment_round_trip(store):
    exp_id = _create_experiments(store, "test_get_ra")
    created = _create_assignment(store, exp_id)

    fetched = store.get_review_assignment(created.assignment_id)

    assert fetched == created


def test_get_review_assignment_raises_on_missing(store):
    with pytest.raises(MlflowException, match="not found"):
        store.get_review_assignment("ra-does-not-exist")


def test_list_review_assignments_filters_by_experiment(store):
    exp_a, exp_b = _create_experiments(store, ["test_list_exp_a", "test_list_exp_b"])
    _create_assignment(store, exp_a, target_id="tr-a", reviewer="sme-a@example.com")
    _create_assignment(store, exp_b, target_id="tr-b", reviewer="sme-b@example.com")

    page = store.list_review_assignments(experiment_id=exp_a)

    assert len(page) == 1
    assert page[0].experiment_id == exp_a
    assert page[0].target_id == "tr-a"


def test_list_review_assignments_filters_by_reviewer_case_insensitively(store):
    exp_id = _create_experiments(store, "test_list_by_reviewer")
    _create_assignment(store, exp_id, target_id="tr-1", reviewer="Alice@Example.com")
    _create_assignment(store, exp_id, target_id="tr-2", reviewer="bob@example.com")

    page = store.list_review_assignments(reviewer="ALICE@example.com")

    assert len(page) == 1
    assert page[0].target_id == "tr-1"
    # Stored as canonical lowercase.
    assert page[0].reviewer == "alice@example.com"


def test_list_review_assignments_filters_by_state(store):
    exp_id = _create_experiments(store, "test_list_by_state")
    a = _create_assignment(store, exp_id, target_id="tr-1")
    _create_assignment(store, exp_id, target_id="tr-2")
    store.update_review_assignment(a.assignment_id, state="complete")

    pending_page = store.list_review_assignments(experiment_id=exp_id, state="pending")
    complete_page = store.list_review_assignments(experiment_id=exp_id, state="complete")

    assert {a.target_id for a in pending_page} == {"tr-2"}
    assert {a.target_id for a in complete_page} == {"tr-1"}


def test_list_review_assignments_requires_scope_predicate(store):
    # No experiment_id, no reviewer -> hard error rather than a
    # full-table scan that hides a perf foot-gun.
    with pytest.raises(MlflowException, match="experiment_id"):
        store.list_review_assignments(state="pending")


def test_list_review_assignments_for_target_returns_all_reviewers(store):
    exp_id = _create_experiments(store, "test_list_for_target")
    _create_assignment(store, exp_id, target_id="tr-1", reviewer="alice@example.com")
    _create_assignment(store, exp_id, target_id="tr-1", reviewer="bob@example.com")
    _create_assignment(store, exp_id, target_id="tr-2", reviewer="alice@example.com")

    rows = store.list_review_assignments_for_target("tr-1")

    assert {r.reviewer for r in rows} == {"alice@example.com", "bob@example.com"}


# ---------------------------------------------------------------------------
# update / delete
# ---------------------------------------------------------------------------


def test_update_review_assignment_transitions_state(store):
    exp_id = _create_experiments(store, "test_update_ra_state")
    a = _create_assignment(store, exp_id)
    assert a.state == ReviewAssignmentState.PENDING
    assert a.completed_time_ms is None

    complete = store.update_review_assignment(a.assignment_id, state="complete")
    assert complete.state == ReviewAssignmentState.COMPLETE
    assert complete.last_update_time_ms >= a.last_update_time_ms
    assert complete.completed_time_ms is not None

    reopened = store.update_review_assignment(a.assignment_id, state="pending")
    assert reopened.state == ReviewAssignmentState.PENDING
    # Reopen clears the completion stamp.
    assert reopened.completed_time_ms is None


def test_update_review_assignment_noop_when_state_unchanged(store):
    exp_id = _create_experiments(store, "test_update_ra_noop")
    a = _create_assignment(store, exp_id)

    same = store.update_review_assignment(a.assignment_id, state="pending")

    # No timestamp churn for a no-op update; the row is byte-identical.
    assert same == a


def test_update_review_assignment_rejects_unknown_state(store):
    exp_id = _create_experiments(store, "test_update_ra_bad_state")
    a = _create_assignment(store, exp_id)

    with pytest.raises(MlflowException, match="state"):
        store.update_review_assignment(a.assignment_id, state="not_a_state")


def test_update_review_assignment_reopen_clears_completed_time(store):
    exp_id = _create_experiments(store, "test_reopen_clears_completed")
    a = _create_assignment(store, exp_id)
    store.update_review_assignment(a.assignment_id, state="complete")

    reopened = store.update_review_assignment(a.assignment_id, state="pending")

    assert reopened.state == ReviewAssignmentState.PENDING
    assert reopened.completed_time_ms is None


def test_update_review_assignment_raises_on_missing(store):
    with pytest.raises(MlflowException, match="not found"):
        store.update_review_assignment("ra-missing", state="complete")


def test_delete_review_assignment_removes_row(store):
    exp_id = _create_experiments(store, "test_delete_ra")
    a = _create_assignment(store, exp_id)

    store.delete_review_assignment(a.assignment_id)

    with pytest.raises(MlflowException, match="not found"):
        store.get_review_assignment(a.assignment_id)


def test_delete_review_assignment_is_idempotent_on_missing(store):
    # No exception on deleting a non-existent assignment; the SDK
    # contract treats delete as idempotent so concurrent deletes
    # don't surprise the UI with a 404.
    store.delete_review_assignment("ra-never-existed")


# ---------------------------------------------------------------------------
# Cascade-on-experiment-delete
# ---------------------------------------------------------------------------


def test_cascade_on_hard_delete_of_parent_experiment(store):
    exp_id = _create_experiments(store, "test_cascade")
    a = _create_assignment(store, exp_id)
    # MLflow's default delete_experiment is a soft-delete (sets
    # lifecycle_stage='deleted'); the FK cascade only fires on hard
    # delete, so we issue one through the raw session.
    with store.ManagedSessionMaker(read_only=False) as session:
        from mlflow.store.tracking.dbmodels.models import SqlExperiment

        session.query(SqlExperiment).filter(SqlExperiment.experiment_id == int(exp_id)).delete()

    with pytest.raises(MlflowException, match="not found"):
        store.get_review_assignment(a.assignment_id)


def test_soft_delete_of_parent_experiment_does_not_cascade(store):
    exp_id = _create_experiments(store, "test_soft_delete_keeps_ra")
    a = _create_assignment(store, exp_id)

    store.delete_experiment(exp_id)

    # Soft-delete preserves the assignment row; the experiment is
    # hidden from default lists but its child data isn't dropped.
    fetched = store.get_review_assignment(a.assignment_id)
    assert fetched.assignment_id == a.assignment_id


# ---------------------------------------------------------------------------
# Entity shape sanity (catches accidental dataclass field reordering)
# ---------------------------------------------------------------------------


def test_to_mlflow_entity_returns_review_assignment(store):
    exp_id = _create_experiments(store, "test_entity_shape")
    a = _create_assignment(store, exp_id)
    assert isinstance(a, ReviewAssignment)
