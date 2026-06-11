"""Review queues for expert trace-review workflows.

A ``ReviewQueue`` is a named bundle of attached items, a set of
questions (label schemas), and a set of assigned users, scoped to an
experiment. Two flavors of the same entity:

- a **user queue** (``name`` = a user, exactly that one user, all of the
  experiment's schemas) is a reviewer's personal worklist; and
- a **custom queue** (arbitrary name, 0..N users, a chosen subset of
  schemas) is a curated, possibly collaborative review task.

Assigned users form a *pool*: an item is addressed when any one of them
acts on it, and the per-``(queue, item)`` :class:`ReviewStatus` records
who. Reviewers answer the queue's questions by writing ``Feedback``
assessments against the item (no new answer storage); the queue carries
only the review *workflow*.

This module exposes the entity types plus the fluent SDK for managing
queues against the MLflow tracking store.
"""

from typing import TYPE_CHECKING, Literal

from mlflow.exceptions import MlflowException
from mlflow.genai.review_queues.review_queues import (
    ReviewItemType,
    ReviewQueue,
    ReviewQueueItem,
    ReviewQueueType,
    ReviewStatus,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracing.client import TracingClient
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    from mlflow.store.entities.paged_list import PagedList

__all__ = [
    "ReviewItemType",
    "ReviewQueue",
    "ReviewQueueItem",
    "ReviewQueueType",
    "ReviewStatus",
    "add_items_to_review_queue",
    "create_review_queue",
    "delete_review_queue",
    "get_or_create_user_queue",
    "get_review_queue",
    "list_review_queue_items",
    "list_review_queues",
    "remove_items_from_review_queue",
    "set_review_queue_item_status",
    "update_review_queue",
]


def _resolve_experiment_id(experiment_id: str | None) -> str:
    if experiment_id is not None:
        return experiment_id
    from mlflow.tracking.fluent import _get_experiment_id

    return _get_experiment_id()


@experimental(version="3.14.0")
def create_review_queue(
    name: str,
    *,
    queue_type: Literal["user", "custom"],
    users: list[str] | None = None,
    schema_ids: list[str] | None = None,
    experiment_id: str | None = None,
) -> ReviewQueue:
    """Create a review queue scoped to an experiment.

    Args:
        name: Queue name, unique within the experiment. For a ``"user"``
            queue this is the user identifier; ``"default"`` is reserved
            case-insensitively (the no-auth default user queue) and rejected
            for custom queues.
        queue_type: ``"user"`` (exactly one assigned user equal to ``name``,
            inherits all of the experiment's label schemas) or ``"custom"``
            (0 to 4 users and an explicit subset of schemas).
        users: Assigned users (at most 4). Derived as ``[name]`` for a user
            queue when omitted; 0 to 4 for a custom queue.
        schema_ids: Attached label-schema ids. Must be empty for a user
            queue (it resolves to all of the experiment's schemas); the chosen
            subset for a custom queue.
        experiment_id: Parent experiment; defaults to the current experiment.

    Returns:
        The created :py:class:`ReviewQueue`. Its owner (``created_by``) is set
        by the server from the authenticated user, not by the caller.
    """
    return TracingClient()._create_review_queue(
        _resolve_experiment_id(experiment_id),
        name=name,
        queue_type=queue_type,
        users=users,
        schema_ids=schema_ids,
    )


@experimental(version="3.14.0")
def get_or_create_user_queue(
    user: str,
    *,
    experiment_id: str | None = None,
) -> ReviewQueue:
    """Return a user's personal review queue, creating it if absent.

    Idempotent: the backbone of "assign these items to this person" — call
    this, then :func:`add_items_to_review_queue`.

    Args:
        user: The reviewer identifier (also the queue name).
        experiment_id: Parent experiment; defaults to the current experiment.

    Returns:
        The user's :py:class:`ReviewQueue` (owned by that user).
    """
    return TracingClient()._get_or_create_user_queue(
        _resolve_experiment_id(experiment_id), user=user
    )


@experimental(version="3.14.0")
def get_review_queue(
    queue_id: str | None = None,
    *,
    name: str | None = None,
    experiment_id: str | None = None,
) -> ReviewQueue:
    """Fetch a review queue by ``queue_id`` or by ``(experiment_id, name)``.

    Provide exactly one of ``queue_id`` or ``name``. When ``name`` is given,
    ``experiment_id`` defaults to the current experiment.

    Returns:
        The matching :py:class:`ReviewQueue`.
    """
    if (queue_id is None) == (name is None):
        raise MlflowException(
            "Provide exactly one of `queue_id` or `name`.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    client = TracingClient()
    if queue_id is not None:
        return client._get_review_queue(queue_id)
    return client._get_review_queue_by_name(_resolve_experiment_id(experiment_id), name)


@experimental(version="3.14.0")
def list_review_queues(
    *,
    user: str | None = None,
    experiment_id: str | None = None,
    max_results: int | None = None,
    page_token: str | None = None,
) -> "PagedList[ReviewQueue]":
    """List an experiment's review queues, newest first.

    Args:
        user: If set, return only queues this user is assigned to.
        experiment_id: Parent experiment; defaults to the current experiment.
        max_results: Page size.
        page_token: Continuation token from a previous call.

    Returns:
        A :py:class:`PagedList` of :py:class:`ReviewQueue`.
    """
    return TracingClient()._list_review_queues(
        _resolve_experiment_id(experiment_id),
        user=user,
        max_results=max_results,
        page_token=page_token,
    )


@experimental(version="3.14.0")
def update_review_queue(
    queue_id: str,
    *,
    name: str | None = None,
    new_owner: str | None = None,
    users: list[str] | None = None,
    schema_ids: list[str] | None = None,
) -> ReviewQueue:
    """Update a custom queue's name, owner, assigned users, and/or schemas.

    Pass only the fields you want to change; ``None`` leaves a field untouched
    (an empty ``users`` / ``schema_ids`` list clears that set). ``queue_type`` is
    immutable and user queues reject this. Reassigning the owner (``new_owner``)
    requires experiment MANAGE — enforced server-side — while the queue's owner
    may make the other edits with EDIT. A queue's ``schema_ids`` (its questions)
    are frozen once it has attached items; detach the items first to edit them.

    Returns:
        The updated :py:class:`ReviewQueue`.
    """
    return TracingClient()._update_review_queue(
        queue_id, name=name, new_owner=new_owner, users=users, schema_ids=schema_ids
    )


@experimental(version="3.14.0")
def delete_review_queue(queue_id: str) -> None:
    """Delete a queue and its associations. No-op if it doesn't exist.

    Reviewer assessments on the queue's items are unaffected.
    """
    TracingClient()._delete_review_queue(queue_id)


@experimental(version="3.14.0")
def add_items_to_review_queue(queue_id: str, *, item_ids: list[str]) -> list[ReviewQueueItem]:
    """Attach items to a queue, returning the resulting queue items.

    Idempotent per item (re-attaching preserves the existing status). The
    returned list covers every requested ``item_id``, in request order.
    """
    return TracingClient()._add_items_to_review_queue(queue_id, item_ids=item_ids)


@experimental(version="3.14.0")
def remove_items_from_review_queue(queue_id: str, *, item_ids: list[str]) -> None:
    """Detach items from a queue. No-op for items not attached."""
    TracingClient()._remove_items_from_review_queue(queue_id, item_ids=item_ids)


@experimental(version="3.14.0")
def list_review_queue_items(
    queue_id: str,
    *,
    status: Literal["pending", "complete", "declined"] | None = None,
    max_results: int | None = None,
    page_token: str | None = None,
) -> "PagedList[ReviewQueueItem]":
    """List a queue's attached items, newest-attached first.

    Args:
        queue_id: The queue to list.
        status: Optional filter on shared-pool status.
        max_results: Page size.
        page_token: Continuation token from a previous call.

    Returns:
        A :py:class:`PagedList` of :py:class:`ReviewQueueItem`.
    """
    return TracingClient()._list_review_queue_items(
        queue_id, status=status, max_results=max_results, page_token=page_token
    )


@experimental(version="3.14.0")
def set_review_queue_item_status(
    queue_id: str,
    *,
    item_id: str,
    status: Literal["pending", "complete", "declined"],
    completed_by: str | None = None,
) -> ReviewQueueItem:
    """Set the shared-pool status of an attached item.

    Moving to ``"complete"`` / ``"declined"`` records ``completed_by``;
    moving back to ``"pending"`` (reopen) clears it. ``completed_by`` is
    required for the terminal states and rejected for ``"pending"``.

    Returns:
        The updated :py:class:`ReviewQueueItem`.
    """
    return TracingClient()._set_review_queue_item_status(
        queue_id, item_id=item_id, status=status, completed_by=completed_by
    )
