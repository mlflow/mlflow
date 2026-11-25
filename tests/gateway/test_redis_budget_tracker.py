import dataclasses
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDuration,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    BudgetWindowType,
    GatewayBudgetPolicy,
)
from mlflow.gateway.budget_tracker import BudgetTracker, SlidingWindowHistory

fakeredis = pytest.importorskip("fakeredis")


def _make_policy(
    budget_policy_id="bp-test",
    budget_amount=100.0,
    duration=None,
    target_scope=BudgetTargetScope.GLOBAL,
    budget_action=BudgetAction.ALERT,
    workspace=None,
    target_value=None,
    window_type=BudgetWindowType.FIXED,
):
    return GatewayBudgetPolicy(
        budget_policy_id=budget_policy_id,
        budget_unit=BudgetUnit.USD,
        budget_amount=budget_amount,
        duration=duration or BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1),
        target_scope=target_scope,
        budget_action=budget_action,
        created_at=0,
        last_updated_at=0,
        workspace=workspace,
        target_value=target_value,
        window_type=window_type,
    )


def _make_tracker():
    from mlflow.gateway.budget_tracker.redis import RedisBudgetTracker

    client = fakeredis.FakeRedis(decode_responses=True)
    return RedisBudgetTracker(_client=client)


def _make_shared_trackers():
    """Two trackers on one Redis, standing in for two gateway processes."""
    from mlflow.gateway.budget_tracker.redis import RedisBudgetTracker

    server = fakeredis.FakeServer()
    return (
        RedisBudgetTracker(_client=fakeredis.FakeRedis(server=server, decode_responses=True)),
        RedisBudgetTracker(_client=fakeredis.FakeRedis(server=server, decode_responses=True)),
    )


def test_redis_tracker_is_budget_tracker():
    tracker = _make_tracker()
    assert isinstance(tracker, BudgetTracker)


def test_record_cost_below_limit():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    newly_exceeded = tracker.record_cost(50.0)
    assert newly_exceeded == []

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 50.0
    assert window.exceeded is False


def test_record_cost_exceeds_threshold():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    newly_exceeded = tracker.record_cost(150.0)
    assert len(newly_exceeded) == 1
    assert newly_exceeded[0].policy.budget_policy_id == "bp-test"

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 150.0
    assert window.exceeded is True


def test_record_cost_exceeds_only_once():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    exceeded1 = tracker.record_cost(150.0)
    assert len(exceeded1) == 1

    exceeded2 = tracker.record_cost(50.0)
    assert exceeded2 == []

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 200.0


def test_record_cost_incremental_exceeding():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    assert tracker.record_cost(60.0) == []
    exceeded = tracker.record_cost(50.0)
    assert len(exceeded) == 1
    assert tracker._get_window_info("bp-test").cumulative_spend == 110.0


def test_should_reject_request_reject():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)])

    tracker.record_cost(150.0)
    exceeded, window = tracker.should_reject_request()
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-test"


def test_should_reject_request_alert_only():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.ALERT)])

    tracker.record_cost(150.0)
    exceeded, window = tracker.should_reject_request()
    assert exceeded is False
    assert window is None


def test_should_reject_request_not_yet():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)])

    tracker.record_cost(50.0)
    exceeded, window = tracker.should_reject_request()
    assert exceeded is False
    assert window is None


def test_refresh_policies_removes_deleted_policy():
    tracker = _make_tracker()
    policy1 = _make_policy(budget_policy_id="bp-1", budget_amount=100.0)
    policy2 = _make_policy(budget_policy_id="bp-2", budget_amount=200.0)
    tracker.refresh_policies([policy1, policy2])
    tracker.record_cost(50.0)

    tracker.refresh_policies([policy1])
    assert tracker._get_window_info("bp-1") is not None
    assert tracker._get_window_info("bp-2") is None


def test_multiple_policies_independent():
    tracker = _make_tracker()
    policy_alert = _make_policy(
        budget_policy_id="bp-alert",
        budget_amount=50.0,
        budget_action=BudgetAction.ALERT,
    )
    policy_reject = _make_policy(
        budget_policy_id="bp-reject",
        budget_amount=100.0,
        budget_action=BudgetAction.REJECT,
    )
    tracker.refresh_policies([policy_alert, policy_reject])

    exceeded = tracker.record_cost(75.0)
    assert len(exceeded) == 1
    assert exceeded[0].policy.budget_policy_id == "bp-alert"

    exceeded, _ = tracker.should_reject_request()
    assert exceeded is False

    tracker.record_cost(30.0)
    exceeded, window = tracker.should_reject_request()
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-reject"


def test_workspace_scoped_cost_recording():
    tracker = _make_tracker()
    policy = _make_policy(
        target_scope=BudgetTargetScope.WORKSPACE,
        workspace="ws1",
        budget_amount=100.0,
    )
    tracker.refresh_policies([policy])

    tracker.record_cost(200.0, workspace="ws2")
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 0.0

    tracker.record_cost(50.0, workspace="ws1")
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 50.0


def test_backfill_spend_sets_cumulative():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    tracker.backfill_spend({"bp-test": 42.5})
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 42.5
    assert window.exceeded is False


def test_backfill_spend_sets_exceeded_when_exceeds():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    tracker.backfill_spend({"bp-test": 150.0})
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 150.0
    assert window.exceeded is True


def test_backfill_spend_sets_exceeded_at_exact_limit():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    tracker.backfill_spend({"bp-test": 100.0})
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 100.0
    assert window.exceeded is True


def test_backfill_spend_nonexistent_is_noop():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy()])
    assert tracker.backfill_spend({"nonexistent-policy": 50.0}) == []


def test_backfill_spend_returns_the_window_it_carried_over_the_limit():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    newly_exceeded = tracker.backfill_spend({"bp-test": 150.0})
    assert [w.policy.budget_policy_id for w in newly_exceeded] == ["bp-test"]
    assert newly_exceeded[0].cumulative_spend == 150.0
    assert newly_exceeded[0].exceeded is True


def test_backfill_spend_returns_nothing_below_the_limit():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    assert tracker.backfill_spend({"bp-test": 42.5}) == []


def test_backfill_spend_reports_a_crossing_only_once():
    # The alert is edge-triggered, so a window already over its limit stays quiet.
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    assert len(tracker.backfill_spend({"bp-test": 150.0})) == 1
    assert tracker.backfill_spend({"bp-test": 175.0}) == []


def test_backfill_spend_stays_quiet_when_record_cost_already_reported():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])
    assert len(tracker.record_cost(120.0)) == 1

    assert tracker.backfill_spend({"bp-test": 150.0}) == []


def test_backfill_spend_uses_max_to_protect_in_process_spend():
    # The window hash is the shared counter record_cost bumps, and the trace history the
    # seed comes from lags behind it. Seeding the lower value would regress the counter.
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])
    assert len(tracker.record_cost(120.0)) == 1

    assert tracker.backfill_spend({"bp-test": 50.0}) == []  # DB is behind
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 120.0
    assert window.exceeded is True

    # A regressed counter would re-cross the limit here and alert a second time.
    assert tracker.record_cost(60.0) == []


def test_backfill_spend_keeps_a_reject_policy_rejecting():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0, budget_action=BudgetAction.REJECT)])
    tracker.record_cost(120.0)
    assert tracker.should_reject_request()[0] is True

    tracker.backfill_spend({"bp-test": 50.0})  # DB is behind
    assert tracker.should_reject_request()[0] is True


def test_refresh_policies_returns_new_windows():
    tracker = _make_tracker()
    policy1 = _make_policy(budget_policy_id="bp-1")
    policy2 = _make_policy(budget_policy_id="bp-2")

    new_windows = tracker.refresh_policies([policy1, policy2])
    assert len(new_windows) == 2
    ids = {w.policy.budget_policy_id for w in new_windows}
    assert ids == {"bp-1", "bp-2"}


def test_refresh_policies_is_idempotent_for_existing_policies():
    tracker = _make_tracker()
    policy = _make_policy(budget_policy_id="bp-1", budget_amount=100.0)

    first_windows = tracker.refresh_policies([policy])
    assert len(first_windows) == 1

    tracker.backfill_spend({"bp-1": 42.5})
    window_before = tracker._get_window_info("bp-1")
    assert window_before.cumulative_spend == 42.5

    # Second call with the same policy should not create a new window
    second_windows = tracker.refresh_policies([policy])
    assert len(second_windows) == 0

    # Existing window state should be preserved
    window_after = tracker._get_window_info("bp-1")
    assert window_after.cumulative_spend == window_before.cumulative_spend
    assert window_after.exceeded == window_before.exceeded


def test_get_all_windows():
    tracker = _make_tracker()
    policy1 = _make_policy(budget_policy_id="bp-1", budget_amount=100.0)
    policy2 = _make_policy(budget_policy_id="bp-2", budget_amount=200.0)
    tracker.refresh_policies([policy1, policy2])

    tracker.record_cost(75.0)

    windows = tracker.get_all_windows()
    assert len(windows) == 2
    by_id = {w.policy.budget_policy_id: w for w in windows}
    assert by_id["bp-1"].cumulative_spend == 75.0
    assert by_id["bp-1"].exceeded is False
    assert by_id["bp-2"].cumulative_spend == 75.0
    assert by_id["bp-2"].exceeded is False


def test_should_reject_request_workspace_filtering():
    tracker = _make_tracker()
    policy = _make_policy(
        target_scope=BudgetTargetScope.WORKSPACE,
        workspace="ws1",
        budget_amount=100.0,
        budget_action=BudgetAction.REJECT,
    )
    tracker.refresh_policies([policy])

    tracker.record_cost(150.0, workspace="ws1")

    exceeded, window = tracker.should_reject_request(workspace="ws2")
    assert exceeded is False
    assert window is None

    exceeded, window = tracker.should_reject_request(workspace="ws1")
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-test"


def test_endpoint_scoped_cost_recording():
    tracker = _make_tracker()
    policy = _make_policy(
        budget_policy_id="bp-ep",
        target_scope=BudgetTargetScope.ENDPOINT,
        target_value="ep-1",
        budget_amount=100.0,
    )
    tracker.refresh_policies([policy])

    tracker.record_cost(200.0, endpoint_id="ep-2")
    assert tracker._get_window_info("bp-ep").cumulative_spend == 0.0

    tracker.record_cost(50.0, endpoint_id="ep-1")
    assert tracker._get_window_info("bp-ep").cumulative_spend == 50.0


def test_should_reject_request_endpoint_filtering():
    tracker = _make_tracker()
    policy = _make_policy(
        budget_policy_id="bp-ep",
        target_scope=BudgetTargetScope.ENDPOINT,
        target_value="ep-1",
        budget_amount=100.0,
        budget_action=BudgetAction.REJECT,
    )
    tracker.refresh_policies([policy])

    tracker.record_cost(150.0, endpoint_id="ep-1")

    exceeded, window = tracker.should_reject_request(endpoint_id="ep-2")
    assert exceeded is False
    assert window is None

    exceeded, window = tracker.should_reject_request(endpoint_id="ep-1")
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-ep"


def test_endpoint_scoped_policy_reloaded_from_redis():
    # Drop the in-process policy cache to force deserialization from Redis and
    # confirm endpoint_id survives the round trip.
    tracker = _make_tracker()
    policy = _make_policy(
        budget_policy_id="bp-ep",
        target_scope=BudgetTargetScope.ENDPOINT,
        target_value="ep-1",
        budget_amount=100.0,
        budget_action=BudgetAction.REJECT,
    )
    tracker.refresh_policies([policy])
    tracker.record_cost(150.0, endpoint_id="ep-1")

    tracker._policy_cache.clear()

    exceeded, window = tracker.should_reject_request(endpoint_id="ep-1")
    assert exceeded is True
    assert window.policy.target_value == "ep-1"

    exceeded_other, _ = tracker.should_reject_request(endpoint_id="ep-2")
    assert exceeded_other is False


def test_record_cost_at_exact_budget_boundary():
    tracker = _make_tracker()
    tracker.refresh_policies([_make_policy(budget_amount=100.0)])

    exceeded = tracker.record_cost(100.0)
    assert len(exceeded) == 1
    assert exceeded[0].policy.budget_policy_id == "bp-test"

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 100.0
    assert window.exceeded is True


def test_window_rollover_resets_spend():
    tracker = _make_tracker()
    tracker.refresh_policies([
        _make_policy(
            budget_amount=100.0, duration=BudgetDuration(unit=BudgetDurationUnit.MINUTES, value=1)
        )
    ])

    tracker.record_cost(150.0)
    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 150.0
    assert window.exceeded is True

    # Simulate time advancing past the window boundary
    future = datetime.now(timezone.utc) + timedelta(minutes=2)
    with patch(
        "mlflow.gateway.budget_tracker.redis.datetime",
    ) as mock_dt:
        mock_dt.now.return_value = future
        mock_dt.fromisoformat = datetime.fromisoformat

        tracker.record_cost(10.0)

    window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 10.0
    assert window.exceeded is False


def test_should_reject_request_ignores_window_from_another_policy_version():
    # A budget edit reaches one process before the other. The process still on the old
    # 1 MONTHS / $500 version rewrites the shared window to monthly bounds, so the
    # process already on 1 DAYS / $100 must not measure them against its daily limit.
    stale_proc, fresh_proc = _make_shared_trackers()
    monthly = _make_policy(
        budget_policy_id="bp-edited",
        budget_amount=500.0,
        duration=BudgetDuration(unit=BudgetDurationUnit.MONTHS, value=1),
    )
    daily = _make_policy(
        budget_policy_id="bp-edited",
        budget_amount=100.0,
        duration=BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1),
        budget_action=BudgetAction.REJECT,
    )

    fresh_proc.refresh_policies([daily])
    stale_proc.refresh_policies([monthly])
    stale_proc.record_cost(150.0)

    exceeded, window = fresh_proc.should_reject_request()
    assert exceeded is False
    assert window is None


def test_should_reject_request_uses_window_once_the_durations_agree():
    stale_proc, fresh_proc = _make_shared_trackers()
    daily = _make_policy(
        budget_policy_id="bp-edited",
        budget_amount=100.0,
        duration=BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1),
        budget_action=BudgetAction.REJECT,
    )

    fresh_proc.refresh_policies([daily])
    stale_proc.refresh_policies([daily])
    stale_proc.record_cost(150.0)

    exceeded, window = fresh_proc.should_reject_request()
    assert exceeded is True
    assert window.cumulative_spend == 150.0


def test_window_rolls_when_duration_changes_without_changing_window_start():
    # On the first of a month, MONTHS and DAYS windows begin at the same instant, so
    # the stored window_start on its own cannot tell the two budget periods apart.
    tracker = _make_tracker()
    first_of_month = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)

    with patch("mlflow.gateway.budget_tracker.redis.datetime") as mock_dt:
        mock_dt.now.return_value = first_of_month
        mock_dt.fromisoformat = datetime.fromisoformat

        tracker.refresh_policies([
            _make_policy(
                budget_amount=500.0,
                duration=BudgetDuration(unit=BudgetDurationUnit.MONTHS, value=1),
            )
        ])
        tracker.record_cost(150.0)

        rolled = tracker.refresh_policies([
            _make_policy(
                budget_amount=100.0,
                duration=BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1),
            )
        ])

    assert len(rolled) == 1
    window = tracker._get_window_info("bp-test")
    assert window.window_start == datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert window.window_end == datetime(2026, 9, 2, tzinfo=timezone.utc)
    assert window.cumulative_spend == 0.0


def test_window_written_without_a_duration_rolls_once():
    # Windows stored before the duration was recorded alongside the bounds are rolled
    # once, after which the caller reseeds them from trace history.
    from mlflow.gateway.budget_tracker.redis import _window_key

    tracker = _make_tracker()
    policy = _make_policy(budget_amount=100.0)
    tracker.refresh_policies([policy])
    tracker.record_cost(50.0)

    tracker._client.hdel(_window_key("bp-test"), "duration_unit", "duration_value")

    rolled = tracker.refresh_policies([policy])
    assert len(rolled) == 1
    assert tracker._get_window_info("bp-test").cumulative_spend == 0.0


def test_get_budget_tracker_returns_redis_when_configured():
    from mlflow.gateway.budget_tracker.redis import RedisBudgetTracker

    with (
        patch(
            "mlflow.gateway.budget_tracker.MLFLOW_GATEWAY_BUDGET_REDIS_URL.get",
            return_value="redis://localhost:6379/0",
        ),
        patch(
            "mlflow.gateway.budget_tracker.redis.RedisBudgetTracker.__post_init__",
        ) as mock_init,
        patch(
            "mlflow.gateway.budget_tracker._budget_tracker",
            new=None,
        ),
    ):
        from mlflow.gateway.budget_tracker import get_budget_tracker

        tracker = get_budget_tracker()
        assert isinstance(tracker, RedisBudgetTracker)
        mock_init.assert_called_once()

        # Reset the singleton
        import mlflow.gateway.budget_tracker

        mlflow.gateway.budget_tracker._budget_tracker = None


# --- USER-scope tests ---


def test_serialize_roundtrip_preserves_target_value():
    from mlflow.gateway.budget_tracker.redis import _deserialize_policy, _serialize_policy

    policy = _make_policy(target_scope=BudgetTargetScope.USER, target_value="alice")
    restored = _deserialize_policy(_serialize_policy(policy))
    assert restored.target_scope == BudgetTargetScope.USER
    assert restored.target_value == "alice"


def test_user_scoped_cost_recording_matches_username():
    tracker = _make_tracker()
    tracker.refresh_policies([
        _make_policy(target_scope=BudgetTargetScope.USER, target_value="alice", budget_amount=100.0)
    ])

    tracker.record_cost(200.0, username="bob")
    assert tracker._get_window_info("bp-test").cumulative_spend == 0.0

    tracker.record_cost(50.0, username="alice")
    assert tracker._get_window_info("bp-test").cumulative_spend == 50.0


def test_user_scoped_should_reject_only_for_matching_username():
    tracker = _make_tracker()
    tracker.refresh_policies([
        _make_policy(
            target_scope=BudgetTargetScope.USER,
            target_value="alice",
            budget_amount=100.0,
            budget_action=BudgetAction.REJECT,
        )
    ])
    tracker.record_cost(150.0, username="alice")

    exceeded, window = tracker.should_reject_request(username="alice")
    assert exceeded is True
    assert window.policy.budget_policy_id == "bp-test"

    exceeded, window = tracker.should_reject_request(username="bob")
    assert exceeded is False
    assert window is None


def test_multiple_user_policies_are_independent():
    tracker = _make_tracker()
    tracker.refresh_policies([
        _make_policy(
            budget_policy_id="bp-alice",
            target_scope=BudgetTargetScope.USER,
            target_value="alice",
            budget_amount=100.0,
        ),
        _make_policy(
            budget_policy_id="bp-bob",
            target_scope=BudgetTargetScope.USER,
            target_value="bob",
            budget_amount=100.0,
        ),
    ])

    exceeded = tracker.record_cost(150.0, username="alice")
    assert {w.policy.budget_policy_id for w in exceeded} == {"bp-alice"}
    assert tracker._get_window_info("bp-alice").cumulative_spend == 150.0
    assert tracker._get_window_info("bp-bob").cumulative_spend == 0.0


# --- Sliding windows ---

_NOW = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)
_ONE_HOUR = BudgetDuration(unit=BudgetDurationUnit.HOURS, value=1)


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


@contextmanager
def _clock(moment: datetime):
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    with patch("mlflow.gateway.budget_tracker.redis.datetime", _FrozenDatetime):
        yield


def _sliding_policy(budget_amount=50.0, budget_action=BudgetAction.REJECT, **kwargs):
    kwargs.setdefault("duration", _ONE_HOUR)
    return _make_policy(
        budget_amount=budget_amount,
        budget_action=budget_action,
        window_type=BudgetWindowType.SLIDING,
        **kwargs,
    )


def test_redis_sliding_window_counts_only_the_trailing_duration():
    tracker = _make_tracker()
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=1000.0)])
        tracker.record_cost(30.0, timestamp_ms=_ms(_NOW - timedelta(minutes=50)))
        tracker.record_cost(40.0, timestamp_ms=_ms(_NOW - timedelta(minutes=10)))
        assert tracker._get_window_info("bp-test").cumulative_spend == 70.0

    with _clock(_NOW + timedelta(minutes=15)):
        window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 40.0
    assert window.window_start == _NOW - timedelta(minutes=45)
    # The aged-out bucket is gone from Redis, not just skipped.
    stored = tracker._client.hgetall("mlflow:budget:window:bp-test")
    assert [f for f in stored if f.startswith("b:")] == [f"b:{_ms(_NOW - timedelta(minutes=10))}"]


def test_redis_sliding_window_ignores_cost_of_request_started_before_the_window():
    tracker = _make_tracker()
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=1000.0)])
        assert tracker.record_cost(10.0, timestamp_ms=_ms(_NOW - timedelta(hours=2))) == []
        assert tracker._get_window_info("bp-test").cumulative_spend == 0.0


def test_redis_sliding_window_rejects_until_enough_spend_ages_out():
    tracker = _make_tracker()
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=50.0)])
        tracker.record_cost(30.0, timestamp_ms=_ms(_NOW - timedelta(minutes=40)))
        tracker.record_cost(30.0, timestamp_ms=_ms(_NOW - timedelta(minutes=5)))
        exceeded, window = tracker.should_reject_request()
    assert exceeded is True
    assert window.cumulative_spend == 60.0
    assert window.resets_at == _NOW + timedelta(minutes=21)

    with _clock(_NOW + timedelta(minutes=21) - timedelta(seconds=1)):
        assert tracker.should_reject_request()[0] is True
    with _clock(_NOW + timedelta(minutes=21)):
        assert tracker.should_reject_request() == (False, None)


def test_redis_sliding_window_alert_rearms_after_spend_drops_below_the_limit():
    tracker = _make_tracker()
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_action=BudgetAction.ALERT)])
        assert len(tracker.record_cost(60.0)) == 1
        assert tracker.record_cost(10.0) == []

    with _clock(_NOW + timedelta(hours=1, minutes=1)):
        newly_exceeded = tracker.record_cost(60.0)
    assert len(newly_exceeded) == 1
    assert newly_exceeded[0].cumulative_spend == 60.0


def test_redis_sliding_window_is_shared_and_crossing_is_reported_once():
    tracker_a, tracker_b = _make_shared_trackers()
    policy = _sliding_policy(budget_amount=50.0, budget_action=BudgetAction.ALERT)
    with _clock(_NOW):
        tracker_a.refresh_policies([policy])
        tracker_b.refresh_policies([policy])
        assert tracker_a.record_cost(30.0) == []
        crossed = tracker_b.record_cost(30.0)
        assert tracker_a.record_cost(30.0) == []
        assert tracker_b._get_window_info("bp-test").cumulative_spend == 90.0
    assert len(crossed) == 1


def test_redis_refresh_returns_sliding_windows_for_backfill_every_time():
    tracker = _make_tracker()
    fixed = _make_policy(budget_policy_id="bp-fixed")
    sliding = _sliding_policy(budget_policy_id="bp-sliding")
    with _clock(_NOW):
        first = tracker.refresh_policies([fixed, sliding])
        second = tracker.refresh_policies([fixed, sliding])
    assert {w.policy.budget_policy_id for w in first} == {"bp-fixed", "bp-sliding"}
    # A fixed window is only seeded when created; a sliding one is synced on every refresh.
    assert [w.policy.budget_policy_id for w in second] == ["bp-sliding"]


def test_redis_sliding_window_recovers_history_missed_by_an_earlier_refresh():
    tracker = _make_tracker()
    policy = _sliding_policy(budget_amount=50.0)
    bucket = _ms(_NOW - timedelta(minutes=10))
    with _clock(_NOW):
        # The first refresh's history query failed, so nothing was seeded.
        tracker.refresh_policies([policy])
        assert tracker.should_reject_request() == (False, None)

        [window] = tracker.refresh_policies([policy])
        tracker.backfill_bucket_spend([
            SlidingWindowHistory(policy=window.policy, bucket_spend={bucket: 60.0})
        ])
        assert tracker.should_reject_request()[0] is True


def test_redis_sliding_window_resets_when_duration_changes():
    tracker = _make_tracker()
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy(budget_amount=1000.0)])
        tracker.record_cost(30.0)
        two_hours = BudgetDuration(unit=BudgetDurationUnit.HOURS, value=2)
        fresh = tracker.refresh_policies([
            _sliding_policy(budget_amount=1000.0, duration=two_hours)
        ])
        assert len(fresh) == 1
        assert tracker._get_window_info("bp-test").cumulative_spend == 0.0


def test_redis_switching_window_type_never_resurrects_old_buckets():
    tracker = _make_tracker()
    sliding = _sliding_policy(budget_amount=1000.0)
    fixed = _make_policy(budget_amount=1000.0, duration=_ONE_HOUR)
    with _clock(_NOW):
        tracker.refresh_policies([sliding])
        tracker.record_cost(30.0)
        tracker.refresh_policies([fixed])
        assert tracker._get_window_info("bp-test").cumulative_spend == 0.0
        tracker.record_cost(5.0)
        assert tracker._get_window_info("bp-test").cumulative_spend == 5.0
        tracker.refresh_policies([sliding])
        window = tracker._get_window_info("bp-test")
    assert window.policy.window_type == BudgetWindowType.SLIDING
    assert window.cumulative_spend == 0.0
    assert window.bucket_spend == {}


def test_redis_stale_process_does_not_reset_a_newer_sliding_window():
    newer, stale = _make_shared_trackers()
    with _clock(_NOW):
        newer.refresh_policies([_sliding_policy(budget_amount=50.0)])
        newer.record_cost(60.0)
        # The stale process still caches the fixed version of this policy.
        stale._policy_cache = {"bp-test": _make_policy(budget_amount=50.0, duration=_ONE_HOUR)}
        stale._client.sadd("mlflow:budget:policies", "bp-test")
        assert stale.should_reject_request() == (False, None)
        assert newer.should_reject_request()[0] is True


def test_redis_sliding_check_with_a_stale_duration_leaves_the_window_alone():
    newer, stale = _make_shared_trackers()
    with _clock(_NOW):
        newer.refresh_policies([_sliding_policy(budget_amount=50.0)])
        newer.record_cost(60.0)
        two_hours = BudgetDuration(unit=BudgetDurationUnit.HOURS, value=2)
        stale._policy_cache = {"bp-test": _sliding_policy(budget_amount=50.0, duration=two_hours)}
        assert stale.should_reject_request() == (False, None)
        assert newer.should_reject_request()[0] is True


def test_redis_backfill_bucket_spend_takes_max_per_bucket_and_reports_crossing_once():
    tracker_a, tracker_b = _make_shared_trackers()
    policy = _sliding_policy(budget_amount=50.0, budget_action=BudgetAction.ALERT)
    bucket_a = _ms(_NOW - timedelta(minutes=30))
    bucket_b = _ms(_NOW - timedelta(minutes=10))
    with _clock(_NOW):
        tracker_a.refresh_policies([policy])
        tracker_b.refresh_policies([policy])
        tracker_a.record_cost(20.0, timestamp_ms=bucket_a)
        seeded = [SlidingWindowHistory(policy=policy, bucket_spend={bucket_a: 5.0, bucket_b: 40.0})]
        crossed = tracker_a.backfill_bucket_spend(seeded)
        assert tracker_b.backfill_bucket_spend(seeded) == []
        window = tracker_b._get_window_info("bp-test")
    assert len(crossed) == 1
    assert window.bucket_spend == {bucket_a: 20.0, bucket_b: 40.0}
    assert window.cumulative_spend == 60.0


def test_redis_backfill_bucket_spend_reports_crossing_after_earlier_spend_aged_out():
    tracker = _make_tracker()
    policy = _sliding_policy(budget_action=BudgetAction.ALERT)
    with _clock(_NOW):
        tracker.refresh_policies([policy])
        assert len(tracker.record_cost(60.0)) == 1

    later = _NOW + timedelta(hours=1, minutes=1)
    with _clock(later):
        crossed = tracker.backfill_bucket_spend([
            SlidingWindowHistory(policy=policy, bucket_spend={_ms(later): 60.0})
        ])
    assert len(crossed) == 1


def test_redis_backfill_methods_ignore_the_other_window_type():
    tracker = _make_tracker()
    with _clock(_NOW):
        tracker.refresh_policies([
            _make_policy(budget_policy_id="bp-fixed"),
            _sliding_policy(budget_policy_id="bp-sliding"),
        ])
        assert tracker.backfill_spend({"bp-sliding": 100.0}) == []
        fixed_history = SlidingWindowHistory(
            policy=_make_policy(budget_policy_id="bp-fixed"), bucket_spend={_ms(_NOW): 100.0}
        )
        assert tracker.backfill_bucket_spend([fixed_history]) == []
        windows = {w.policy.budget_policy_id: w for w in tracker.get_all_windows()}
    assert windows["bp-fixed"].cumulative_spend == 0.0
    assert windows["bp-sliding"].cumulative_spend == 0.0


def test_redis_serialize_round_trips_window_type():
    from mlflow.gateway.budget_tracker.redis import _deserialize_policy, _serialize_policy

    restored = _deserialize_policy(_serialize_policy(_sliding_policy()))
    assert restored.window_type == BudgetWindowType.SLIDING


def test_redis_deserialize_policy_written_before_sliding_windows_is_fixed():
    from mlflow.gateway.budget_tracker.redis import _deserialize_policy, _serialize_policy

    data = json.loads(_serialize_policy(_make_policy()))
    del data["window_type"]
    assert _deserialize_policy(json.dumps(data)).window_type == BudgetWindowType.FIXED


# --- Policy revisions: processes that have not picked up an edit yet ---


def _revised(policy: GatewayBudgetPolicy, revision: int, **changes) -> GatewayBudgetPolicy:
    return dataclasses.replace(policy, last_updated_at=revision, **changes)


def test_redis_stale_process_records_into_the_newer_sliding_window():
    newer, stale = _make_shared_trackers()
    v1 = _revised(_sliding_policy(budget_amount=1000.0), 1)
    v2 = _revised(v1, 2, duration=BudgetDuration(unit=BudgetDurationUnit.HOURS, value=2))
    with _clock(_NOW):
        stale.refresh_policies([v1])
        newer.refresh_policies([v2])
        newer.record_cost(30.0)
        # Still on v1, whose 1-hour config would reset the window and wipe the $30.
        stale.record_cost(10.0)
        window = newer._get_window_info("bp-test")
    assert window.cumulative_spend == 40.0
    assert window.policy.duration == v2.duration
    assert stale._policy_cache["bp-test"].last_updated_at == 2


def test_redis_stale_limit_does_not_clear_the_alert_flag():
    newer, stale = _make_shared_trackers()
    v1 = _revised(_sliding_policy(budget_amount=100.0, budget_action=BudgetAction.ALERT), 1)
    v2 = _revised(v1, 2, budget_amount=50.0)
    with _clock(_NOW):
        stale.refresh_policies([v1])
        newer.refresh_policies([v2])
        assert len(newer.record_cost(60.0)) == 1
        # Judged against v1's $100 limit, $70 would clear the flag and the next request
        # would alert a second time.
        assert stale.record_cost(10.0) == []
        assert newer.record_cost(10.0) == []


def test_redis_stale_fixed_process_does_not_replace_a_sliding_window():
    newer, stale = _make_shared_trackers()
    v1 = _revised(_make_policy(budget_amount=1000.0, duration=_ONE_HOUR), 1)
    v2 = _revised(v1, 2, window_type=BudgetWindowType.SLIDING)
    with _clock(_NOW):
        stale.refresh_policies([v1])
        newer.refresh_policies([v2])
        newer.record_cost(30.0)
        stale.record_cost(10.0)
        window = newer._get_window_info("bp-test")
    assert window.policy.window_type == BudgetWindowType.SLIDING
    assert window.cumulative_spend == 40.0


def test_redis_stale_process_rejects_with_the_newer_limit():
    newer, stale = _make_shared_trackers()
    v1 = _revised(_sliding_policy(budget_amount=100.0), 1)
    v2 = _revised(v1, 2, budget_amount=50.0)
    with _clock(_NOW):
        stale.refresh_policies([v1])
        newer.refresh_policies([v2])
        newer.record_cost(60.0)
        exceeded, window = stale.should_reject_request()
    assert exceeded is True
    assert window.policy.budget_amount == 50.0


def test_redis_backfill_drops_history_read_for_an_older_policy():
    tracker = _make_tracker()
    v1 = _revised(_sliding_policy(budget_amount=1000.0), 1)
    v2 = _revised(v1, 2, duration=BudgetDuration(unit=BudgetDurationUnit.HOURS, value=2))
    with _clock(_NOW):
        tracker.refresh_policies([v2])
        tracker.record_cost(30.0)
        stale_history = SlidingWindowHistory(
            policy=v1, bucket_spend={_ms(_NOW - timedelta(minutes=1)): 500.0}
        )
        assert tracker.backfill_bucket_spend([stale_history]) == []
        window = tracker._get_window_info("bp-test")
    assert window.cumulative_spend == 30.0
    assert window.policy.duration == v2.duration


def test_redis_refresh_does_not_regress_a_newer_stored_policy():
    newer, stale = _make_shared_trackers()
    v1 = _revised(_sliding_policy(budget_amount=100.0), 1)
    v2 = _revised(v1, 2, budget_amount=50.0)
    with _clock(_NOW):
        newer.refresh_policies([v2])
        # A refresh that read the database before the edit lands afterwards.
        stale.refresh_policies([v1])
    from mlflow.gateway.budget_tracker.redis import _deserialize_policy

    stored = _deserialize_policy(newer._client.hget("mlflow:budget:policy:bp-test", "data"))
    assert stored.last_updated_at == 2
    assert stored.budget_amount == 50.0


def test_redis_sliding_window_expires_after_its_newest_bucket_ages_out():
    tracker = _make_tracker()
    with _clock(_NOW):
        tracker.refresh_policies([_sliding_policy()])
        tracker.record_cost(10.0)
    ttl = tracker._client.ttl("mlflow:budget:window:bp-test")
    # The newest bucket (12:00-12:01) ages out at 13:01, plus an hour of grace.
    assert 61 * 60 + 3600 - 60 < ttl <= 61 * 60 + 3600
