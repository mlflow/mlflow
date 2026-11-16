"""Checkpoint (watermark) computation shared by the trace- and session-level processors.

Both processors discover entities (traces or sessions) ordered by ``(timestamp_ms, id)`` and
persist a checkpoint of the same shape. The checkpoint is a durable watermark: every entity at
or before it has either been durably assessed, deterministically sampled out, or given up on
after a bounded wait. It must therefore never move past an entity that was not fetched (the page
limit was hit) or that was fetched but not durably processed. :func:`compute_checkpoint`
implements that rule in one place.
"""

from dataclasses import dataclass
from typing import Iterable

# (timestamp_ms, entity_id). ``entity_id`` is ``None`` for a checkpoint that means "everything up
# to and including this timestamp is done, nothing at a later timestamp has been processed yet".
EntityKey = tuple[int, str | None]


@dataclass
class ConsideredEntity:
    """A trace or session that was fetched in this run, and whether it is durably done."""

    timestamp_ms: int
    entity_id: str
    done: bool


def _sort_key(key: EntityKey) -> tuple[int, str]:
    return (key[0], key[1] or "")


def is_after(key: EntityKey, boundary: EntityKey) -> bool:
    """Whether ``key`` sorts strictly after ``boundary`` in ``(timestamp_ms, id)`` order."""
    return _sort_key(key) > _sort_key(boundary)


def safe_watermark(full_page_last_keys: Iterable[EntityKey]) -> EntityKey | None:
    """
    Compute the highest key that every discovery query in this run is guaranteed to have covered.

    Discovery runs one query per filter group, each capped at the per-job page size. A group whose
    page was full may have more matching entities beyond its last row, so nothing after that row
    can be considered covered. The watermark is the smallest such last row across all full pages;
    ``None`` means no page was full and the whole window was covered.
    """
    keys = list(full_page_last_keys)
    if not keys:
        return None
    return min(keys, key=_sort_key)


def compute_checkpoint(
    *,
    previous: EntityKey | None,
    window_max_ms: int,
    considered: Iterable[ConsideredEntity],
    watermark: EntityKey | None,
) -> EntityKey | None:
    """
    Compute the new checkpoint for a run, or ``None`` if it must not advance.

    Args:
        previous: The checkpoint at the start of the run, if any.
        window_max_ms: Upper bound of the discovery window for this run.
        considered: Every entity fetched in this run with its ``done`` status. Entities that
            sort after ``watermark`` are treated as not done regardless of their flag, since they
            will be re-discovered by the next run.
        watermark: Result of :func:`safe_watermark` for this run.

    Returns:
        The new checkpoint, or ``None`` when it would not move forward. The rules are:

        * If any considered entity is not done, the checkpoint stops just before the earliest
          such entity, at ``(timestamp - 1, None)``. Using a bare timestamp rather than the
          preceding entity's id avoids comparing ids in Python, whose ordering can differ from
          the store's collation. Done entities sharing that millisecond are re-discovered next
          run, which is safe because processing is idempotent.
        * Otherwise, if a page was full, the checkpoint is the watermark itself (a key returned
          by the store, so consistent with the store's ordering).
        * Otherwise every entity in the window was covered and the checkpoint is the window end.
    """
    not_done_ts = [
        e.timestamp_ms
        for e in considered
        if not e.done
        or (watermark is not None and is_after((e.timestamp_ms, e.entity_id), watermark))
    ]

    if not_done_ts:
        candidate: EntityKey = (min(not_done_ts) - 1, None)
        # Stopping before the earliest pending entity is only safe up to the watermark: a full
        # page may hide unfetched entities between the watermark and that entity.
        if watermark is not None and is_after(candidate, watermark):
            candidate = watermark
    elif watermark is not None:
        candidate = watermark
    else:
        candidate = (window_max_ms, None)

    if previous is not None and not is_after(candidate, previous):
        return None
    return candidate
