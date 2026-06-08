"""Server-side validation and normalization for review queues.

Called from the store layer's create / update / attach / status paths.
Length caps on the validated identity fields (queue name, user, schema id,
target id) are aligned with their SQL column widths so a value that passes
validation also fits its column. Audit-only fields populated by the server
(e.g. ``created_by``) are not part of the validated payload and follow the
same convention as the sibling ``label_schemas`` store (unvalidated).

This module is store-layer-internal; callers should not import it
directly.
"""

from __future__ import annotations

from typing import NamedTuple

from mlflow.exceptions import MlflowException
from mlflow.genai.review_queues.review_queues import (
    ReviewQueueType,
    ReviewStatus,
    ReviewTargetType,
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
# `target_id` mirrors `SqlReviewQueueTrace.target_id` (`String(50)`).
TARGET_ID_MAX_LENGTH = 50

# Reserved for the legacy no-auth user queue; custom queues may not use it.
RESERVED_QUEUE_NAME = "default"

# Name of the experiment's single default queue (a special CUSTOM queue that
# inherits all of the experiment's schemas and is undeletable). Reserved from
# regular custom queues.
DEFAULT_QUEUE_NAME = "Default"


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


def coerce_target_type(target_type: object) -> ReviewTargetType:
    if isinstance(target_type, ReviewTargetType):
        return target_type
    if target_type not in ReviewTargetType:
        raise _invalid(
            f"`target_type` must be one of {ReviewTargetType.values()}; got {target_type!r}."
        )
    return ReviewTargetType(target_type)


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


def normalize_target_id(target_id: object) -> str:
    """Strip a target identifier.

    Trace ids are case-sensitive (UUID/hex), so only strip whitespace.
    """
    if not isinstance(target_id, str):
        raise _invalid(f"`target_id` must be a string; got {target_id!r}.")
    return target_id.strip()


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
    return _dedup_preserving_order(normalized)


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
    is_default: bool = False


def validate_queue_for_create(
    *,
    name: object,
    queue_type: object,
    users: list[str] | None,
    schema_ids: list[str] | None,
    is_default: bool = False,
) -> ValidatedQueue:
    """Validate + normalize a queue about to be created, enforcing the
    user-vs-custom invariants.

    User queue: ``name`` is normalized as a user identifier, the assigned
    set is exactly that one user (derived when ``users`` is omitted), and no
    schemas may be attached (a user queue resolves to all of the
    experiment's schemas at read time). Custom queue: ``name`` is stripped,
    case-preserved, and may not be a reserved name; users are 0..N and schemas
    are the chosen subset. Default queue (``is_default``): a CUSTOM queue named
    :data:`DEFAULT_QUEUE_NAME` that, like a user queue, attaches no schemas and
    resolves to all of the experiment's schemas at read time.

    ``experiment_id`` is intentionally NOT validated here — the store-layer
    existence check inside the write transaction owns that.
    """
    coerced_type = coerce_queue_type(queue_type)
    normalized_users = normalize_users(users)
    normalized_schema_ids = normalize_schema_ids(schema_ids)

    if is_default:
        if coerced_type != ReviewQueueType.CUSTOM:
            raise _invalid("The default queue must be a custom queue.")
        if normalized_schema_ids:
            raise _invalid(
                "The default queue cannot have explicitly-attached schemas; it resolves to "
                "all of the experiment's label schemas."
            )
        normalized_name = DEFAULT_QUEUE_NAME
    elif coerced_type == ReviewQueueType.USER:
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
        if not isinstance(name, str):
            raise _invalid(f"`name` must be a string; got {name!r}.")
        normalized_name = name.strip()
        _validate_non_empty_string(normalized_name, "name", QUEUE_NAME_MAX_LENGTH)
        if normalized_name in (RESERVED_QUEUE_NAME, DEFAULT_QUEUE_NAME):
            raise _invalid(
                f"`{normalized_name}` is a reserved queue name and cannot be used for a "
                "custom queue."
            )

    return ValidatedQueue(
        queue_type=coerced_type,
        name=normalized_name,
        users=normalized_users,
        schema_ids=normalized_schema_ids,
        is_default=is_default,
    )


def validate_target_ids_for_attach(target_ids: object) -> list[str]:
    """Normalize + validate a batch of target ids for attach/detach.

    Validates the whole batch up front (fail-fast): one bad id fails the
    call rather than silently dropping that row. De-duplicates so attaching
    the same trace twice in one call is a single row.
    """
    if not isinstance(target_ids, (list, tuple)) or len(target_ids) == 0:
        raise _invalid(f"`target_ids` must be a non-empty list; got {target_ids!r}.")
    normalized = []
    for target_id in target_ids:
        normalized_target_id = normalize_target_id(target_id)
        _validate_non_empty_string(normalized_target_id, "target_id", TARGET_ID_MAX_LENGTH)
        normalized.append(normalized_target_id)
    return _dedup_preserving_order(normalized)
