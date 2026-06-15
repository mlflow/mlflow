"""Server-side validation and normalization for review queues.

Called from the store layer's create / update / attach / status paths.
Length caps on the validated identity fields (queue name, user, schema id,
item id) are aligned with their SQL column widths so a value that passes
validation also fits its column. The queue owner (``created_by``) is
server-controlled — stamped from the authenticated user on create — so it is
not part of the create payload validated here.

This module is store-layer-internal; callers should not import it
directly.
"""

from __future__ import annotations

from typing import NamedTuple

from mlflow.exceptions import MlflowException
from mlflow.genai.review_queues.review_queues import (
    ReviewItemType,
    ReviewQueueType,
    ReviewStatus,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

# `name` mirrors `SqlLabelSchema.name` (also the assessment key width):
# queue names and label-schema names share the 250-char ceiling.
QUEUE_NAME_MAX_LENGTH = 250
# `user` mirrors `SqlAssessments.source_id` so an assigned user can never
# be too long to also appear as an assessment `source_id` (a reviewer's
# answers are matched to their queue by this identifier).
USER_MAX_LENGTH = 250
# `schema_id` mirrors `SqlLabelSchema.schema_id` (`String(36)`).
SCHEMA_ID_MAX_LENGTH = 36
# `item_id` mirrors `SqlReviewQueueItem.item_id` (`String(50)`).
ITEM_ID_MAX_LENGTH = 50

# Reserved for the no-auth default user queue; custom queues may not use it
# (case-insensitively, so "Default"/"DEFAULT" are rejected too).
RESERVED_QUEUE_NAME = "default"

# Cap on a queue's assigned users so a queue can't be assigned an unbounded
# reviewer set. A user queue always stays well under this (exactly 1). Keep in
# sync with `MAX_ASSIGNED_USERS` in the frontend (ReviewerChecklistCombobox.tsx),
# which gates the picker so the cap isn't hit on submit.
MAX_ASSIGNED_USERS = 10


def _invalid(message: str) -> MlflowException:
    return MlflowException(message, error_code=INVALID_PARAMETER_VALUE)


def _validate_non_empty_string(value: object, field: str, max_length: int) -> None:
    if not isinstance(value, str) or len(value) == 0:
        raise _invalid(f"`{field}` must be a non-empty string; got {value!r}.")
    if len(value) > max_length:
        raise _invalid(f"`{field}` must be at most {max_length} characters; got {len(value)}.")


def coerce_queue_type(queue_type: object) -> ReviewQueueType:
    if isinstance(queue_type, ReviewQueueType):
        return queue_type
    if queue_type not in ReviewQueueType:
        raise _invalid(
            f"`queue_type` must be one of {ReviewQueueType.values()}; got {queue_type!r}."
        )
    return ReviewQueueType(queue_type)


def coerce_item_type(item_type: object) -> ReviewItemType:
    if isinstance(item_type, ReviewItemType):
        return item_type
    if item_type not in ReviewItemType:
        raise _invalid(f"`item_type` must be one of {ReviewItemType.values()}; got {item_type!r}.")
    return ReviewItemType(item_type)


def coerce_status(status: object) -> ReviewStatus:
    if isinstance(status, ReviewStatus):
        return status
    if status not in ReviewStatus:
        raise _invalid(f"`status` must be one of {ReviewStatus.values()}; got {status!r}.")
    return ReviewStatus(status)


def normalize_user(user: object) -> str:
    """Lowercase + strip a user identifier.

    Identity comparisons against ``AssessmentSource.source_id`` are
    case-insensitive, so we normalize once at write time and store the
    canonical form. This lets the assigned-user set and (for user queues)
    the queue ``name`` match a reviewer's assessments without casing drift.
    """
    if not isinstance(user, str):
        raise _invalid(f"`user` must be a string; got {user!r}.")
    return user.strip().lower()


def normalize_item_id(item_id: object) -> str:
    """Strip an item identifier.

    Trace ids are case-sensitive (UUID/hex), so only strip whitespace.
    """
    if not isinstance(item_id, str):
        raise _invalid(f"`item_id` must be a string; got {item_id!r}.")
    return item_id.strip()


def normalize_schema_id(schema_id: object) -> str:
    """Strip a label-schema identifier (case-sensitive surrogate id)."""
    if not isinstance(schema_id, str):
        raise _invalid(f"`schema_id` must be a string; got {schema_id!r}.")
    return schema_id.strip()


def _dedup_preserving_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def normalize_users(users: list[str] | None) -> list[str]:
    """Normalize + validate + de-duplicate an assigned-user list."""
    if users is None:
        return []
    if not isinstance(users, (list, tuple)):
        raise _invalid(f"`users` must be a list of strings; got {users!r}.")
    normalized = []
    for user in users:
        normalized_user = normalize_user(user)
        _validate_non_empty_string(normalized_user, "user", USER_MAX_LENGTH)
        normalized.append(normalized_user)
    deduped = _dedup_preserving_order(normalized)
    if len(deduped) > MAX_ASSIGNED_USERS:
        raise _invalid(
            f"A review queue can have at most {MAX_ASSIGNED_USERS} assigned users; "
            f"got {len(deduped)}."
        )
    return deduped


def normalize_schema_ids(schema_ids: list[str] | None) -> list[str]:
    """Normalize + validate + de-duplicate an attached-schema-id list."""
    if schema_ids is None:
        return []
    if not isinstance(schema_ids, (list, tuple)):
        raise _invalid(f"`schema_ids` must be a list of strings; got {schema_ids!r}.")
    normalized = []
    for schema_id in schema_ids:
        normalized_schema_id = normalize_schema_id(schema_id)
        _validate_non_empty_string(normalized_schema_id, "schema_id", SCHEMA_ID_MAX_LENGTH)
        normalized.append(normalized_schema_id)
    return _dedup_preserving_order(normalized)


class ValidatedQueue(NamedTuple):
    """A normalized, invariant-checked queue ready to insert."""

    queue_type: ReviewQueueType
    name: str
    users: list[str]
    schema_ids: list[str]


def validate_custom_queue_name(name: object) -> str:
    """Validate + normalize a CUSTOM queue name: a stripped, case-preserved,
    non-empty, non-reserved string. Used by the queue-create path.
    """
    if not isinstance(name, str):
        raise _invalid(f"`name` must be a string; got {name!r}.")
    normalized_name = name.strip()
    _validate_non_empty_string(normalized_name, "name", QUEUE_NAME_MAX_LENGTH)
    if normalized_name.lower() == RESERVED_QUEUE_NAME:
        raise _invalid(
            f"`{normalized_name}` is a reserved queue name and cannot be used for a custom queue."
        )
    return normalized_name


def validate_queue_for_create(
    *,
    name: object,
    queue_type: object,
    users: list[str] | None,
    schema_ids: list[str] | None,
) -> ValidatedQueue:
    """Validate + normalize a queue about to be created, enforcing the
    user-vs-custom invariants.

    User queue: ``name`` is normalized as a user identifier, the assigned
    set is exactly that one user (derived when ``users`` is omitted), and no
    schemas may be attached (a user queue resolves to all of the
    experiment's schemas at read time). Custom queue: ``name`` is stripped,
    case-preserved, and may not be a reserved name; users are 0..N and schemas
    are the chosen subset.

    ``experiment_id`` is intentionally NOT validated here — the store-layer
    existence check inside the write transaction owns that.
    """
    coerced_type = coerce_queue_type(queue_type)
    normalized_users = normalize_users(users)
    normalized_schema_ids = normalize_schema_ids(schema_ids)

    if coerced_type == ReviewQueueType.USER:
        normalized_name = normalize_user(name)
        _validate_non_empty_string(normalized_name, "name", QUEUE_NAME_MAX_LENGTH)
        if not normalized_users:
            normalized_users = [normalized_name]
        elif normalized_users != [normalized_name]:
            raise _invalid(
                "A user queue must have exactly one assigned user equal to its name; "
                f"got name={normalized_name!r} and users={normalized_users!r}."
            )
        if normalized_schema_ids:
            raise _invalid(
                "A user queue cannot have explicitly-attached schemas; it resolves to all "
                "of the experiment's label schemas."
            )
    else:
        normalized_name = validate_custom_queue_name(name)

    return ValidatedQueue(
        queue_type=coerced_type,
        name=normalized_name,
        users=normalized_users,
        schema_ids=normalized_schema_ids,
    )


def validate_item_ids_for_attach(item_ids: object) -> list[str]:
    """Normalize + validate a batch of item ids for attach/detach.

    Validates the whole batch up front (fail-fast): one bad id fails the
    call rather than silently dropping that row. De-duplicates so attaching
    the same item twice in one call is a single row.
    """
    if not isinstance(item_ids, (list, tuple)) or len(item_ids) == 0:
        raise _invalid(f"`item_ids` must be a non-empty list; got {item_ids!r}.")
    normalized = []
    for item_id in item_ids:
        normalized_item_id = normalize_item_id(item_id)
        _validate_non_empty_string(normalized_item_id, "item_id", ITEM_ID_MAX_LENGTH)
        normalized.append(normalized_item_id)
    return _dedup_preserving_order(normalized)
