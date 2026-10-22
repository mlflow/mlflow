import json
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

from mlflow.entities.experiment import Experiment
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.genai.discovery.checkpoint import FIRST_RUN_LOOKBACK_MS
from mlflow.genai.discovery.schedule import (
    MAX_TRACE_FILTERS,
    SCHEDULE_VERSION,
    DetectionCadence,
    DetectionSchedule,
    DetectionScheduleState,
    ScheduleTraceFilter,
    claim_first_occurrence,
    claim_occurrence,
    due_occurrence_ms,
    next_fire_time_ms,
    parse_detection_schedule,
    parse_detection_schedule_state,
    parse_trace_filters,
    read_detection_schedule,
    read_detection_schedule_state,
    scheduled_detection_available,
    write_detection_schedule,
    write_detection_schedule_state,
)
from mlflow.utils.mlflow_tags import (
    MLFLOW_ISSUE_DETECTION_SCHEDULE,
    MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE,
    MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER,
)

HOUR_MS = 60 * 60 * 1000
UTC = ZoneInfo("UTC")


def _experiment(tags: dict[str, str]) -> Experiment:
    return Experiment(
        experiment_id="1",
        name="exp",
        artifact_location="loc",
        lifecycle_stage="active",
        tags=[ExperimentTag(key, value) for key, value in tags.items()],
    )


def _ms(zone: str, *args: int) -> int:
    return int(datetime(*args, tzinfo=ZoneInfo(zone)).timestamp() * 1000)


def _local(zone: str, timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=ZoneInfo(zone))


def test_schedule_round_trips():
    schedule = DetectionSchedule(
        cadence=DetectionCadence.WEEKLY,
        timezone="Asia/Tokyo",
        paused=True,
        trace_filters=(
            ScheduleTraceFilter(field="metadata", key="environment", value="production"),
            ScheduleTraceFilter(field="tag", key="team", value="search"),
        ),
    )
    parsed = parse_detection_schedule(schedule.to_json(), schedule.trace_filters_to_json())
    assert parsed == schedule


def test_serialized_envelopes_match_the_stored_shape():
    schedule = DetectionSchedule(
        cadence=DetectionCadence.DAILY,
        timezone="Europe/Berlin",
        trace_filters=(
            ScheduleTraceFilter(field="metadata", key="environment", value="production"),
        ),
    )
    assert json.loads(schedule.to_json()) == {
        "version": SCHEDULE_VERSION,
        "cadence": "daily",
        "timezone": "Europe/Berlin",
        "paused": False,
    }
    assert json.loads(schedule.trace_filters_to_json()) == [
        {"field": "metadata", "operator": "=", "key": "environment", "value": "production"}
    ]


@pytest.mark.parametrize(
    "raw_value",
    [
        None,
        "",
        "not json",
        "[]",
        json.dumps({"cadence": "daily"}),
        json.dumps({"version": SCHEDULE_VERSION + 1, "cadence": "daily"}),
        json.dumps({"version": SCHEDULE_VERSION, "cadence": "monthly"}),
        json.dumps({"version": SCHEDULE_VERSION, "cadence": "off"}),
        json.dumps({"version": SCHEDULE_VERSION}),
    ],
)
def test_unusable_envelopes_read_as_no_schedule(raw_value: str | None):
    assert parse_detection_schedule(raw_value) is None


def test_a_missing_timezone_falls_back_to_utc():
    parsed = parse_detection_schedule(json.dumps({"version": SCHEDULE_VERSION, "cadence": "daily"}))
    assert parsed.timezone == "UTC"


def test_a_timezone_this_server_cannot_load_falls_back_to_utc():
    parsed = parse_detection_schedule(
        json.dumps({
            "version": SCHEDULE_VERSION,
            "cadence": "daily",
            "timezone": "Mars/Olympus_Mons",
        })
    )
    assert parsed.timezone == "UTC"
    assert parsed.cadence is DetectionCadence.DAILY


@pytest.mark.parametrize("paused", ["true", 1, None, "yes"])
def test_only_a_boolean_true_pauses_a_schedule(paused: object):
    parsed = parse_detection_schedule(
        json.dumps({"version": SCHEDULE_VERSION, "cadence": "daily", "paused": paused})
    )
    assert parsed.paused is False


def test_filter_clauses_name_the_right_namespace():
    metadata = ScheduleTraceFilter(field="metadata", key="environment", value="production")
    tag = ScheduleTraceFilter(field="tag", key="team", value="search")
    assert metadata.to_filter_clause() == "request_metadata.`environment` = 'production'"
    assert tag.to_filter_clause() == "tags.`team` = 'search'"


def test_several_clauses_are_anded():
    schedule = DetectionSchedule(
        cadence=DetectionCadence.DAILY,
        trace_filters=(
            ScheduleTraceFilter(field="metadata", key="environment", value="production"),
            ScheduleTraceFilter(field="tag", key="team", value="search"),
        ),
    )
    assert schedule.to_filter_string() == (
        "request_metadata.`environment` = 'production' AND tags.`team` = 'search'"
    )


def test_no_clauses_make_an_empty_filter_string():
    assert DetectionSchedule(cadence=DetectionCadence.DAILY).to_filter_string() == ""


@pytest.mark.parametrize(
    "entry",
    [
        {"field": "span", "key": "k", "value": "v"},
        {"field": "metadata", "key": "", "value": "v"},
        {"field": "metadata", "key": 7, "value": "v"},
        {"field": "metadata", "key": "k", "value": None},
        {"field": "metadata", "key": "k", "value": "v", "operator": "!="},
        {"field": "metadata", "key": "k`", "value": "v"},
        {"field": "metadata", "key": "k", "value": "v' OR '1' = '1"},
        {"field": "metadata", "key": "k", "value": 'say "hi"'},
        "not an object",
        None,
    ],
)
def test_a_clause_this_server_cannot_apply_is_dropped(entry: object):
    assert parse_trace_filters(json.dumps([entry])) == ()


def test_a_dropped_clause_does_not_discard_the_usable_ones():
    filters = parse_trace_filters(
        json.dumps([
            {"field": "metadata", "key": "environment", "value": "production"},
            {"field": "metadata", "key": "bad`key", "value": "v"},
            {"field": "tag", "key": "team", "value": "search"},
        ])
    )
    assert filters == (
        ScheduleTraceFilter(field="metadata", key="environment", value="production"),
        ScheduleTraceFilter(field="tag", key="team", value="search"),
    )


@pytest.mark.parametrize("raw_value", [None, "", "not json", '{"field": "metadata"}'])
def test_an_unreadable_filter_tag_reads_as_no_clauses(raw_value: str | None):
    assert parse_trace_filters(raw_value) == ()


def test_more_clauses_than_the_cap_are_truncated_not_dropped():
    entries = [
        {"field": "metadata", "key": f"k{index}", "value": "v"}
        for index in range(MAX_TRACE_FILTERS + 5)
    ]
    assert len(parse_trace_filters(json.dumps(entries))) == MAX_TRACE_FILTERS


def test_schedule_state_round_trips():
    state = DetectionScheduleState(
        last_fire_time_ms=1_800_000_000_000,
        last_dispatched_at_ms=1_800_000_060_000,
        last_run_id="run-1",
    )
    assert parse_detection_schedule_state(state.to_json()) == state


@pytest.mark.parametrize(
    "raw_value",
    [
        None,
        "not json",
        json.dumps({"lastFireTimeMs": 1}),
        json.dumps({"version": SCHEDULE_VERSION}),
        json.dumps({"version": SCHEDULE_VERSION, "lastFireTimeMs": "soon"}),
        json.dumps({"version": SCHEDULE_VERSION, "lastFireTimeMs": True}),
    ],
)
def test_unusable_envelopes_read_as_no_schedule_state(raw_value: str | None):
    assert parse_detection_schedule_state(raw_value) is None


def test_read_and_write_use_the_experiment_tags():
    schedule = DetectionSchedule(
        cadence=DetectionCadence.HOURLY,
        trace_filters=(ScheduleTraceFilter(field="tag", key="team", value="search"),),
    )
    state = DetectionScheduleState(last_fire_time_ms=1_800_000_000_000)
    experiment = _experiment({
        MLFLOW_ISSUE_DETECTION_SCHEDULE: schedule.to_json(),
        MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER: schedule.trace_filters_to_json(),
        MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE: state.to_json(),
    })
    assert read_detection_schedule(experiment) == schedule
    assert read_detection_schedule_state(experiment) == state
    assert read_detection_schedule(_experiment({})) is None
    assert read_detection_schedule_state(_experiment({})) is None

    store = mock.Mock()
    write_detection_schedule(store, "1", schedule)
    write_detection_schedule_state(store, "1", state)
    written = {tag.key: tag.value for (_, tag), _ in [(c.args, c.kwargs) for c in store.mock_calls]}
    assert set(written) == {
        MLFLOW_ISSUE_DETECTION_SCHEDULE,
        MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER,
        MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE,
    }
    assert (
        parse_detection_schedule(
            written[MLFLOW_ISSUE_DETECTION_SCHEDULE],
            written[MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER],
        )
        == schedule
    )


def test_saving_a_schedule_without_clauses_clears_a_stale_filter_tag():
    store = mock.Mock()
    write_detection_schedule(store, "1", DetectionSchedule(cadence=DetectionCadence.DAILY))
    filter_tags = [
        call.args[1]
        for call in store.set_experiment_tag.call_args_list
        if call.args[1].key == MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER
    ]
    assert [tag.value for tag in filter_tags] == ["[]"]


def test_hourly_fires_on_the_hour():
    schedule = DetectionSchedule(cadence=DetectionCadence.HOURLY)
    after = _ms("UTC", 2026, 5, 4, 13, 37)
    assert _local("UTC", next_fire_time_ms(schedule, after)) == datetime(2026, 5, 4, 14, tzinfo=UTC)


def test_a_fire_time_exactly_at_the_anchor_is_not_returned_again():
    schedule = DetectionSchedule(cadence=DetectionCadence.HOURLY)
    on_the_hour = _ms("UTC", 2026, 5, 4, 14)
    assert next_fire_time_ms(schedule, on_the_hour) == _ms("UTC", 2026, 5, 4, 15)


def test_daily_fires_at_nine_in_the_schedule_timezone():
    schedule = DetectionSchedule(cadence=DetectionCadence.DAILY, timezone="Asia/Tokyo")
    after = _ms("Asia/Tokyo", 2026, 5, 4, 10)
    fired = _local("Asia/Tokyo", next_fire_time_ms(schedule, after))
    assert (fired.hour, fired.minute, fired.day) == (9, 0, 5)


def test_daily_fires_later_today_when_nine_has_not_passed():
    schedule = DetectionSchedule(cadence=DetectionCadence.DAILY, timezone="Asia/Tokyo")
    after = _ms("Asia/Tokyo", 2026, 5, 4, 8)
    fired = _local("Asia/Tokyo", next_fire_time_ms(schedule, after))
    assert (fired.hour, fired.day) == (9, 4)


def test_weekly_fires_monday_at_nine():
    schedule = DetectionSchedule(cadence=DetectionCadence.WEEKLY, timezone="Europe/Berlin")
    # 2026-05-06 is a Wednesday.
    after = _ms("Europe/Berlin", 2026, 5, 6, 12)
    fired = _local("Europe/Berlin", next_fire_time_ms(schedule, after))
    assert (fired.weekday(), fired.hour, fired.day) == (0, 9, 11)


def test_weekly_fires_later_today_when_monday_nine_has_not_passed():
    schedule = DetectionSchedule(cadence=DetectionCadence.WEEKLY, timezone="Europe/Berlin")
    # 2026-05-04 is a Monday.
    after = _ms("Europe/Berlin", 2026, 5, 4, 7)
    fired = _local("Europe/Berlin", next_fire_time_ms(schedule, after))
    assert (fired.weekday(), fired.hour, fired.day) == (0, 9, 4)


def test_daily_keeps_its_local_hour_across_a_spring_forward():
    schedule = DetectionSchedule(cadence=DetectionCadence.DAILY, timezone="America/New_York")
    # US clocks go forward on 2026-03-08.
    after = _ms("America/New_York", 2026, 3, 7, 10)
    fired = _local("America/New_York", next_fire_time_ms(schedule, after))
    assert (fired.day, fired.hour, fired.utcoffset().total_seconds()) == (8, 9, -4 * 3600)


def test_daily_keeps_its_local_hour_across_a_fall_back():
    schedule = DetectionSchedule(cadence=DetectionCadence.DAILY, timezone="America/New_York")
    # US clocks go back on 2026-11-01.
    after = _ms("America/New_York", 2026, 10, 31, 10)
    fired = _local("America/New_York", next_fire_time_ms(schedule, after))
    assert (fired.day, fired.hour, fired.utcoffset().total_seconds()) == (1, 9, -5 * 3600)


def test_hourly_fires_on_the_local_hour_in_a_sub_hour_offset_zone():
    schedule = DetectionSchedule(cadence=DetectionCadence.HOURLY, timezone="Asia/Kathmandu")
    after = _ms("Asia/Kathmandu", 2026, 5, 4, 13, 37)
    fired = _local("Asia/Kathmandu", next_fire_time_ms(schedule, after))
    assert (fired.hour, fired.minute) == (14, 0)


def test_a_schedule_the_server_has_not_seen_is_not_due_yet():
    schedule = DetectionSchedule(cadence=DetectionCadence.HOURLY)
    assert due_occurrence_ms(schedule, None, _ms("UTC", 2026, 5, 4, 14, 1)) is None


def test_the_first_occurrence_is_the_next_fire_time_after_the_server_noticed_the_schedule():
    schedule = DetectionSchedule(cadence=DetectionCadence.DAILY, timezone="UTC")
    noticed = _ms("UTC", 2026, 5, 4, 10)
    state = claim_first_occurrence(noticed)
    assert state == DetectionScheduleState(last_fire_time_ms=noticed)
    # Today's 09:00 already passed when the schedule was saved, so it does not run on the spot.
    assert due_occurrence_ms(schedule, state, noticed + 60_000) is None
    assert due_occurrence_ms(schedule, state, _ms("UTC", 2026, 5, 5, 9)) == _ms(
        "UTC", 2026, 5, 5, 9
    )


def test_a_paused_schedule_is_never_due():
    schedule = DetectionSchedule(cadence=DetectionCadence.HOURLY, paused=True)
    state = DetectionScheduleState(last_fire_time_ms=_ms("UTC", 2026, 5, 4, 13))
    assert due_occurrence_ms(schedule, state, _ms("UTC", 2026, 5, 4, 23)) is None


def test_a_dispatched_occurrence_is_not_due_a_second_time():
    schedule = DetectionSchedule(cadence=DetectionCadence.HOURLY)
    state = DetectionScheduleState(last_fire_time_ms=_ms("UTC", 2026, 5, 4, 13))
    fire_time = due_occurrence_ms(schedule, state, _ms("UTC", 2026, 5, 4, 14, 0, 30))
    assert fire_time == _ms("UTC", 2026, 5, 4, 14)

    claimed = claim_occurrence(state, fire_time, now_ms=_ms("UTC", 2026, 5, 4, 14, 0, 30))
    assert claimed.last_fire_time_ms == fire_time
    assert due_occurrence_ms(schedule, claimed, _ms("UTC", 2026, 5, 4, 14, 30)) is None


def test_downtime_spanning_many_fire_times_resumes_with_one_occurrence():
    schedule = DetectionSchedule(cadence=DetectionCadence.HOURLY)
    state = DetectionScheduleState(last_fire_time_ms=_ms("UTC", 2026, 5, 1, 13))
    back_up = _ms("UTC", 2026, 5, 4, 14, 5)
    fire_time = due_occurrence_ms(schedule, state, back_up)
    # The oldest missed occurrence, not a backlog of every hour in between.
    assert fire_time == _ms("UTC", 2026, 5, 1, 14)
    claimed = claim_occurrence(state, fire_time, now_ms=back_up)
    assert claimed.last_dispatched_at_ms == back_up
    # And the next pass picks up the following one, one occurrence at a time.
    assert due_occurrence_ms(schedule, claimed, back_up) == _ms("UTC", 2026, 5, 1, 15)


def test_the_first_run_lookback_matches_the_cadences_a_schedule_offers():
    assert set(FIRST_RUN_LOOKBACK_MS) == {cadence.value for cadence in DetectionCadence}


@pytest.mark.parametrize(
    ("job_execution", "periodic_tasks", "requirements_met", "expected"),
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_availability_needs_jobs_periodic_tasks_and_a_sql_backend(
    monkeypatch: pytest.MonkeyPatch,
    job_execution: bool,
    periodic_tasks: bool,
    requirements_met: bool,
    expected: bool,
):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", str(job_execution).lower())
    monkeypatch.setenv("MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS", str(periodic_tasks).lower())
    side_effect = None if requirements_met else Exception("no SQL backend store")
    with mock.patch(
        "mlflow.server.jobs.utils._check_requirements", side_effect=side_effect
    ) as check_requirements:
        assert scheduled_detection_available() is expected
    assert check_requirements.called is (job_execution and periodic_tasks)
