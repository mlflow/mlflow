"""Server-side validation and normalization for review assignments.

Called from the store layer's create / bulk-create / update paths.
Length caps are aligned with the SQL column widths so a payload that
passes validation also fits the table (no silent truncation).

This module is store-layer-internal; callers should not import it
directly.
"""

from __future__ import annotations

from mlflow.exceptions import MlflowException
from mlflow.genai.review_assignments.review_assignments import (
    ReviewAssignmentState,
    ReviewTargetType,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

# Aligned with `SqlReviewAssignment` column widths. The 250 mirrors
# `SqlAssessments.source_id`: the UI surfaces a reviewer's own
# assessments alongside their assignment, and keeping the two columns
# the same width means a `reviewer` value can never be too long to also
# appear as an assessment `source_id`.
REVIEWER_MAX_LENGTH = 250
ASSIGNER_MAX_LENGTH = 250
TARGET_ID_MAX_LENGTH = 50

# Cap on the bulk-create cross product (`len(target_ids) * len(reviewers)`).
# Mirrors `MAX_ENTITIES_PER_BATCH` for log-batch: a single write
# transaction shouldn't fan out unboundedly, and the response carries the
# full entity per created row, so an unbounded N*M would blow the gRPC
# message ceiling. Callers needing more should chunk and call repeatedly;
# idempotency on the unique key makes overlap safe.
MAX_BULK_CREATE_ASSIGNMENTS = 1000


def _invalid(message: str) -> MlflowException:
    return MlflowException(message, error_code=INVALID_PARAMETER_VALUE)


def validate_bulk_create_size(num_targets: int, num_reviewers: int) -> None:
    total = num_targets * num_reviewers
    if total > MAX_BULK_CREATE_ASSIGNMENTS:
        raise _invalid(
            f"bulk-create requests at most {MAX_BULK_CREATE_ASSIGNMENTS} assignments "
            f"(len(target_ids) * len(reviewers)); got {num_targets} * {num_reviewers} "
            f"= {total}. Chunk the request and call repeatedly."
        )


def _validate_non_empty_string(value: object, field: str, max_length: int) -> None:
    if not isinstance(value, str) or len(value) == 0:
        raise _invalid(f"`{field}` must be a non-empty string; got {value!r}.")
    if len(value) > max_length:
        raise _invalid(f"`{field}` must be at most {max_length} characters; got {len(value)}.")


def _coerce_target_type(target_type: object) -> ReviewTargetType:
    if isinstance(target_type, ReviewTargetType):
        return target_type
    if target_type not in ReviewTargetType:
        raise _invalid(
            f"`target_type` must be one of {ReviewTargetType.values()}; got {target_type!r}."
        )
    return ReviewTargetType(target_type)


def _validate_state(state: object) -> None:
    if isinstance(state, ReviewAssignmentState):
        return
    if state not in ReviewAssignmentState:
        raise _invalid(f"`state` must be one of {ReviewAssignmentState.values()}; got {state!r}.")


def normalize_reviewer(reviewer: object) -> str:
    """Lowercase + strip a reviewer identifier.

    Identity comparisons against ``AssessmentSource.source_id`` are
    case-insensitive, so we normalize once at write time and store the
    canonical form. This lets the UNIQUE constraint on raw
    ``reviewer`` actually enforce one-row-per-reviewer (lower-casing
    on the read side leaves the constraint case-preserving and allows
    duplicates to race in) and lets indexed lookups use the column
    directly.
    """
    if not isinstance(reviewer, str):
        raise _invalid(f"`reviewer` must be a string; got {reviewer!r}.")
    return reviewer.strip().lower()


def normalize_target_id(target_id: object) -> str:
    """Strip a target identifier.

    Trace ids are case-sensitive (UUID/hex), so only strip whitespace.
    """
    if not isinstance(target_id, str):
        raise _invalid(f"`target_id` must be a string; got {target_id!r}.")
    return target_id.strip()


def normalize_assigner(assigner: object) -> str:
    """Strip an assigner identifier.

    ``assigner`` is audit data; we preserve casing so the UI can show
    it as the user typed, but trim whitespace so a stray newline from
    a clipboard paste doesn't show up in the audit trail.
    """
    if not isinstance(assigner, str):
        raise _invalid(f"`assigner` must be a string; got {assigner!r}.")
    return assigner.strip()


def validate_assignment_for_create(
    *,
    target_type: object,
    target_id: str,
    reviewer: str,
    assigner: str,
) -> ReviewTargetType:
    """Validate a single assignment about to be inserted.

    Called per-row by both the single-create and bulk-create paths;
    bulk-create runs this on each tuple before opening the transaction
    so a partial bad payload fails fast. Returns the coerced
    ``ReviewTargetType`` so the caller doesn't have to re-coerce.

    ``experiment_id`` is intentionally NOT validated here — the
    store-layer ``_validate_experiment_exists`` check that fires
    inside the write transaction covers both "exists" and "valid id
    shape" and produces the canonical error code mapping.
    """
    coerced_target_type = _coerce_target_type(target_type)
    _validate_non_empty_string(target_id, "target_id", TARGET_ID_MAX_LENGTH)
    _validate_non_empty_string(reviewer, "reviewer", REVIEWER_MAX_LENGTH)
    _validate_non_empty_string(assigner, "assigner", ASSIGNER_MAX_LENGTH)
    return coerced_target_type


def validate_assignment_for_update(*, new_state: object) -> ReviewAssignmentState:
    """Validate + coerce a workflow state update.

    Returns the coerced ``ReviewAssignmentState`` so the caller can
    write it directly. With two states (``pending`` / ``complete``)
    every transition is legal, so this only validates that ``new_state``
    is a known state. Only ``state`` is mutable today; identity and
    audit fields are immutable. If/when additional mutable fields are
    added, extend this validator and the store's
    ``update_review_assignment`` accordingly.
    """
    _validate_state(new_state)
    return (
        ReviewAssignmentState(new_state)
        if not isinstance(new_state, ReviewAssignmentState)
        else new_state
    )
