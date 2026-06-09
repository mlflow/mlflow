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

The proto / REST / SDK surface lands in a later stack; this stack ships
the entity, validation, and SQL store.
"""

from mlflow.genai.review_queues.review_queues import (
    ReviewItemType,
    ReviewQueue,
    ReviewQueueItem,
    ReviewQueueType,
    ReviewStatus,
)

__all__ = [
    "ReviewItemType",
    "ReviewQueue",
    "ReviewQueueItem",
    "ReviewQueueType",
    "ReviewStatus",
]
