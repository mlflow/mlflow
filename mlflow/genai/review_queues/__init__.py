"""OSS-native review queues for SME trace-review workflows.

A ``ReviewQueue`` is a named bundle of attached traces, a set of
questions (label schemas), and a set of assigned users, scoped to an
experiment. Two flavors of the same entity:

- a **user queue** (``name`` = a user, exactly that one user, all of the
  experiment's schemas) is a reviewer's personal worklist; and
- a **custom queue** (arbitrary name, 0..N users, a chosen subset of
  schemas) is a curated, possibly collaborative review task.

Assigned users form a *pool*: a trace is addressed when any one of them
acts on it, and the per-``(queue, trace)`` :class:`ReviewStatus` records
who. Reviewers answer the queue's questions by writing ``Feedback``
assessments against the trace (no new answer storage); the queue carries
only the review *workflow*.

The proto / REST / SDK surface lands in a later stack; this stack ships
the entity, validation, and SQL store.
"""

from mlflow.genai.review_queues.review_queues import (
    ReviewQueue,
    ReviewQueueItem,
    ReviewQueueType,
    ReviewStatus,
    ReviewTargetType,
)

__all__ = [
    "ReviewQueue",
    "ReviewQueueItem",
    "ReviewQueueType",
    "ReviewStatus",
    "ReviewTargetType",
]
