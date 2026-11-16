import pytest

from mlflow.genai.scorers.online.watermark import (
    ConsideredEntity,
    compute_checkpoint,
    is_after,
    safe_watermark,
)


def _entity(ts: int, entity_id: str, done: bool = True) -> ConsideredEntity:
    return ConsideredEntity(timestamp_ms=ts, entity_id=entity_id, done=done)


@pytest.mark.parametrize(
    ("key", "boundary", "expected"),
    [
        ((1001, "a"), (1000, "z"), True),
        ((1000, "b"), (1000, "a"), True),
        ((1000, "a"), (1000, "a"), False),
        ((1000, "a"), (1000, "b"), False),
        # A bare-timestamp key sorts before every id at that timestamp.
        ((1000, None), (1000, "a"), False),
        ((1000, "a"), (1000, None), True),
        ((1000, None), (1000, None), False),
        ((999, "z"), (1000, None), False),
    ],
)
def test_is_after(key, boundary, expected):
    assert is_after(key, boundary) is expected


def test_safe_watermark_is_none_when_no_page_was_full():
    assert safe_watermark([]) is None


def test_safe_watermark_is_lowest_full_page_last_key():
    assert safe_watermark([(2000, "b"), (1000, "z"), (1000, "y")]) == (1000, "y")


def test_checkpoint_advances_to_window_end_when_everything_was_covered_and_done():
    considered = [_entity(1000, "a"), _entity(1500, "b")]
    assert compute_checkpoint(
        previous=None, window_max_ms=2000, considered=considered, watermark=None
    ) == (2000, None)


def test_checkpoint_advances_to_window_end_when_nothing_was_found():
    assert compute_checkpoint(previous=None, window_max_ms=2000, considered=[], watermark=None) == (
        2000,
        None,
    )


def test_checkpoint_is_bounded_by_watermark_when_a_page_was_full():
    considered = [_entity(1000, "a"), _entity(1500, "b")]
    assert compute_checkpoint(
        previous=None, window_max_ms=2000, considered=considered, watermark=(1500, "b")
    ) == (1500, "b")


def test_entities_after_watermark_are_pending_even_if_marked_done():
    # Group A's page was full and ended at (1000, "a"); an entity from group B at 1500 was
    # fetched, but group A may still have unfetched entities between 1000 and 1500, so the
    # checkpoint must stop before 1500.
    considered = [_entity(1000, "a"), _entity(1500, "b", done=True)]
    assert compute_checkpoint(
        previous=None, window_max_ms=2000, considered=considered, watermark=(1000, "a")
    ) == (1000, "a")

    # Same, but group B's entity shares the watermark timestamp: its id ordering relative to the
    # watermark decides whether it is covered.
    considered = [_entity(1000, "a"), _entity(1000, "b")]
    assert compute_checkpoint(
        previous=None, window_max_ms=2000, considered=considered, watermark=(1000, "a")
    ) == (999, None)


def test_checkpoint_stops_before_earliest_not_done_entity():
    considered = [_entity(1000, "a"), _entity(1500, "b", done=False), _entity(1800, "c")]
    assert compute_checkpoint(
        previous=None, window_max_ms=2000, considered=considered, watermark=None
    ) == (1499, None)


def test_checkpoint_does_not_move_backwards_or_stay_put():
    # Not-done entity shares the previous checkpoint's millisecond: (999, None) would be a
    # regression, so the checkpoint is left alone.
    considered = [_entity(1000, "b", done=False)]
    assert (
        compute_checkpoint(
            previous=(1000, "a"), window_max_ms=2000, considered=considered, watermark=None
        )
        is None
    )
    # Window end equal to the previous checkpoint: nothing to persist.
    assert (
        compute_checkpoint(previous=(2000, None), window_max_ms=2000, considered=[], watermark=None)
        is None
    )
    # Window end earlier than the previous checkpoint (e.g. buffer was increased): no regression.
    assert (
        compute_checkpoint(previous=(2500, "x"), window_max_ms=2000, considered=[], watermark=None)
        is None
    )


def test_checkpoint_with_bare_timestamp_previous_advances_within_same_millisecond():
    considered = [_entity(1000, "a")]
    assert compute_checkpoint(
        previous=(1000, None), window_max_ms=1000, considered=considered, watermark=(1000, "a")
    ) == (1000, "a")
