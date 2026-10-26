import json
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

from mlflow.entities.experiment import Experiment
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.genai.discovery.checkpoint import FIRST_RUN_LOOKBACK_MS, STALE_WATERMARK_CAP_MS
from mlflow.genai.discovery.schedule import (
    MAX_MODEL_IDENTIFIER_LENGTH,
    MAX_TRACE_FILTERS,
    SCHEDULE_VERSION,
    DetectionCadence,
    DetectionSchedule,
    DetectionScheduleState,
    ScheduleModel,
    ScheduleTraceFilter,
    _latest_fire_time_ms,
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
        {"field": "metadata", "key": "", "value": "v"},
        {"field": "metadata", "key": "   ", "value": "v"},
        {"field": "metadata", "key": 7, "value": "v"},
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


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        # Read the way the tab reads them rather than dropped: an unknown or missing namespace is
        # request metadata, a missing or non-text value is empty, and a padded key is trimmed.
        ({"key": "env", "value": "prod"}, "request_metadata.`env` = 'prod'"),
        ({"field": "span", "key": "env", "value": "prod"}, "request_metadata.`env` = 'prod'"),
        ({"field": "tag", "key": "  team  ", "value": "search"}, "tags.`team` = 'search'"),
        ({"field": "tag", "key": "team"}, "tags.`team` = ''"),
        ({"field": "tag", "key": "team", "value": None}, "tags.`team` = ''"),
        ({"field": "tag", "key": "team", "value": 7}, "tags.`team` = ''"),
    ],
)
def test_a_clause_the_tab_would_keep_is_kept(entry: object, expected: str):
    parsed = parse_trace_filters(json.dumps([entry]))
    assert len(parsed) == 1
    assert parsed[0].to_filter_clause() == expected


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


def test_the_filters_are_stored_before_the_cadence_activates_them():
    # Order is the safety property: the cadence tag is what makes the dispatcher pick a schedule
    # up, so a narrowed scope has to be in force before anything can run under it.
    store = mock.Mock()
    write_detection_schedule(
        store,
        "1",
        DetectionSchedule(
            cadence=DetectionCadence.HOURLY,
            trace_filters=(ScheduleTraceFilter(field="tag", key="team", value="search"),),
        ),
    )
    assert [call.args[1].key for call in store.set_experiment_tag.call_args_list] == [
        MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER,
        MLFLOW_ISSUE_DETECTION_SCHEDULE,
    ]


def test_a_failed_cadence_write_leaves_no_schedule_running_on_the_old_filters():
    # The two writes are not atomic. Failing on the second leaves the narrowed filters stored and
    # the cadence untouched, so nothing runs under the old, wider scope.
    stored: dict[str, str] = {}

    def set_experiment_tag(experiment_id: str, tag: ExperimentTag) -> None:
        if tag.key == MLFLOW_ISSUE_DETECTION_SCHEDULE:
            raise RuntimeError("tag write failed")
        stored[tag.key] = tag.value

    store = mock.Mock()
    store.set_experiment_tag.side_effect = set_experiment_tag
    narrowed = DetectionSchedule(
        cadence=DetectionCadence.HOURLY,
        trace_filters=(ScheduleTraceFilter(field="tag", key="team", value="search"),),
    )
    with pytest.raises(RuntimeError, match="tag write failed"):
        write_detection_schedule(store, "1", narrowed)

    assert MLFLOW_ISSUE_DETECTION_SCHEDULE not in stored
    assert parse_trace_filters(stored[MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER]) == (
        narrowed.trace_filters
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


@pytest.mark.parametrize(
    ("cadence", "latest_fire_time"),
    [
        (DetectionCadence.HOURLY, (2026, 5, 4, 14)),
        (DetectionCadence.DAILY, (2026, 5, 4, 9)),
        (DetectionCadence.WEEKLY, (2026, 5, 4, 9)),
    ],
)
@pytest.mark.parametrize("last_fire_time", [(2026, 5, 1, 13), (2025, 5, 1, 13)])
def test_downtime_resumes_with_one_occurrence_at_the_most_recent_fire_time(
    cadence: DetectionCadence,
    latest_fire_time: tuple[int, ...],
    last_fire_time: tuple[int, ...],
):
    # Drained rather than polled once, so "one run" is the loop's result and not an assumption:
    # claiming the oldest missed occurrence would leave the next poll another one to claim, and a
    # long gap would dispatch a run per interval. Both a three-day and a year-long gap, since the
    # cost of getting this wrong grows with the gap.
    schedule = DetectionSchedule(cadence=cadence)
    state = DetectionScheduleState(last_fire_time_ms=_ms("UTC", *last_fire_time))
    back_up = _ms("UTC", 2026, 5, 4, 14, 5)

    dispatched = []
    while (fire_time := due_occurrence_ms(schedule, state, back_up)) is not None:
        dispatched.append(fire_time)
        state = claim_occurrence(state, fire_time, now_ms=back_up)
        if len(dispatched) > 100:
            break

    # The most recent occurrence due, not the oldest missed: the interval the server came back in.
    assert dispatched == [_ms("UTC", *latest_fire_time)]
    assert state.last_dispatched_at_ms == back_up
    # Nothing is left behind it, so a poll a minute later dispatches nothing.
    assert due_occurrence_ms(schedule, state, back_up + 60_000) is None
    # The claimed occurrence is recent, so the window's stale cap is measured near now rather than
    # near the start of the gap.
    assert back_up - state.last_fire_time_ms < STALE_WATERMARK_CAP_MS


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
    with (
        mock.patch("mlflow.server.jobs.utils.get_job_execution_engine", return_value="huey"),
        mock.patch(
            "mlflow.server.jobs.utils._check_requirements", side_effect=side_effect
        ) as check_requirements,
    ):
        assert scheduled_detection_available() is expected
    assert check_requirements.called is (job_execution and periodic_tasks)


@pytest.mark.parametrize(("engine", "expected"), [("huey", True), ("executor", False)])
def test_the_executor_engine_cannot_run_a_schedule(
    monkeypatch: pytest.MonkeyPatch, engine: str, expected: bool
):
    # That engine rejects the environment a submission carries, which is how a scheduled run
    # receives its model's credential -- so a schedule naming a provider could never authenticate.
    # Reported here rather than discovered one failed occurrence at a time.
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
    monkeypatch.setenv("MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS", "true")
    with (
        mock.patch("mlflow.server.jobs.utils.get_job_execution_engine", return_value=engine),
        mock.patch("mlflow.server.jobs.utils._check_requirements"),
    ):
        assert scheduled_detection_available() is expected


def test_a_model_round_trips_through_the_schedule_envelope():
    schedule = DetectionSchedule(
        cadence=DetectionCadence.DAILY,
        model=ScheduleModel(provider="openai", model="gpt-5.6-sol"),
    )
    assert parse_detection_schedule(schedule.to_json()) == schedule


def test_an_endpoint_round_trips_through_the_schedule_envelope():
    schedule = DetectionSchedule(
        cadence=DetectionCadence.DAILY, model=ScheduleModel(endpoint_name="prod-gateway")
    )
    assert parse_detection_schedule(schedule.to_json()).model == ScheduleModel(
        endpoint_name="prod-gateway"
    )


def test_a_schedule_with_no_model_serializes_without_the_field():
    serialized = json.loads(DetectionSchedule(cadence=DetectionCadence.DAILY).to_json())
    assert "model" not in serialized


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("openai", "gpt-5.6-sol", "openai:/gpt-5.6-sol"),
        # The manual path lowercases the provider before building the URI; so does this.
        ("OpenAI", "gpt-5.6-sol", "openai:/gpt-5.6-sol"),
        ("openrouter", "meta-llama/llama-4", "openrouter:/meta-llama/llama-4"),
    ],
)
def test_the_model_uri_matches_what_the_manual_path_builds(
    provider: str, model: str, expected: str
):
    assert ScheduleModel(provider=provider, model=model).to_model_uri() == expected


def test_an_endpoint_takes_precedence_over_a_provider_in_the_same_envelope():
    # Both halves stored (the picker keeps a provider around while an endpoint is selected): the
    # endpoint is what was chosen, so it is what runs.
    parsed = parse_detection_schedule(
        json.dumps({
            "version": SCHEDULE_VERSION,
            "cadence": "daily",
            "model": {"endpointName": "prod-gateway", "provider": "openai", "model": "gpt-5.6-sol"},
        })
    )
    assert parsed.model.to_model_uri() == "gateway:/prod-gateway"


@pytest.mark.parametrize(
    "model_value",
    [
        None,
        {},
        "openai:/gpt-5.6-sol",
        ["openai", "gpt-5.6-sol"],
        {"provider": "openai"},
        {"model": "gpt-5.6-sol"},
        {"provider": "openai", "model": ""},
        {"provider": "", "model": "gpt-5.6-sol"},
        {"provider": "openai", "model": "   "},
        {"provider": 7, "model": "gpt-5.6-sol"},
        {"provider": True, "model": "gpt-5.6-sol"},
        # Whitespace or control characters would make a model URI nothing names.
        {"provider": "openai", "model": "gpt 5.6"},
        {"provider": "open ai", "model": "gpt-5.6-sol"},
        {"provider": "openai", "model": "a" * (MAX_MODEL_IDENTIFIER_LENGTH + 1)},
        {"endpointName": ""},
        {"endpointName": "prod gateway"},
        {"endpointName": 7},
    ],
)
def test_an_unusable_model_reads_as_no_model_rather_than_failing_the_schedule(model_value: object):
    parsed = parse_detection_schedule(
        json.dumps({"version": SCHEDULE_VERSION, "cadence": "daily", "model": model_value})
    )
    # The schedule still fires -- on the server's default model -- rather than silently stopping.
    assert parsed is not None
    assert parsed.cadence is DetectionCadence.DAILY
    assert parsed.model is None


def test_surrounding_whitespace_is_trimmed_from_a_stored_identifier():
    parsed = parse_detection_schedule(
        json.dumps({
            "version": SCHEDULE_VERSION,
            "cadence": "daily",
            "model": {"provider": " openai ", "model": "gpt-5.6-sol\n"},
        })
    )
    assert parsed.model == ScheduleModel(provider="openai", model="gpt-5.6-sol")


def _brute_force_latest_fire_time_ms(
    schedule: DetectionSchedule, at_or_before_ms: int
) -> int | None:
    """Walk real fire times from well before the target and keep the last one at or before it."""
    cursor = at_or_before_ms - 20 * 24 * 60 * 60 * 1000
    latest = None
    while (nxt := next_fire_time_ms(schedule, cursor)) <= at_or_before_ms:
        latest = nxt
        cursor = nxt
    return latest


@pytest.mark.parametrize(
    ("zone", "transition_utc"),
    [
        # Clocks go back: the wall clock repeats an hour, so a naive candidate is ambiguous.
        ("Europe/London", (2026, 10, 25, 1)),
        ("America/New_York", (2026, 11, 1, 6)),
        # Clocks go forward: the wall clock skips an hour, so a candidate may not exist at all.
        ("Europe/London", (2026, 3, 29, 1)),
        # A 30-minute shift rather than an hour, which an hour-granular candidate set can miss.
        ("Australia/Lord_Howe", (2026, 4, 4, 15, 30)),
    ],
)
@pytest.mark.parametrize(
    "cadence", [DetectionCadence.HOURLY, DetectionCadence.DAILY, DetectionCadence.WEEKLY]
)
def test_the_latest_fire_time_is_found_across_a_dst_transition(zone, transition_utc, cadence):
    # The candidate set is a fixed window of offsets around the reference, so a DST shift that
    # moves a candidate relative to the reference could put the real latest fire time outside it.
    # Swept against a brute-force walk rather than asserted at one instant, since which offset is
    # the right answer changes through the transition.
    schedule = DetectionSchedule(cadence=cadence, timezone=zone)
    transition_ms = _ms("UTC", *transition_utc)
    for offset_minutes in range(-360, 361, 10):
        at_or_before_ms = transition_ms + offset_minutes * 60_000
        assert _latest_fire_time_ms(schedule, at_or_before_ms) == _brute_force_latest_fire_time_ms(
            schedule, at_or_before_ms
        ), f"mismatch {offset_minutes:+d} min from the {zone} transition"


def test_a_trace_filter_carrying_a_control_character_is_dropped():
    # The key and value are spliced into a filter expression, so a newline or a null byte would
    # build a malformed clause.
    for key, value in [
        ("team\x00x", "search"),
        ("team\nx", "search"),
        ("a\tb", "search"),
        ("team", "a\x00b"),
        ("team", "a\nb"),
    ]:
        raw = json.dumps([{"field": "tag", "key": key, "value": value}])
        assert parse_trace_filters(raw) == ()
    # An ordinary space is not a control character and stays usable.
    kept = parse_trace_filters(
        json.dumps([{"field": "tag", "key": "team", "value": "north america"}])
    )
    assert kept == (ScheduleTraceFilter(field="tag", key="team", value="north america"),)


@pytest.mark.parametrize(
    ("stored", "expected_secret_id"),
    [
        ({"provider": "openai", "model": "gpt-5", "secretId": "s-abc"}, "s-abc"),
        # Absent on a schedule saved before the id was recorded, or blank once trimmed: the
        # occurrence falls back to the server's own environment rather than hunting for a key.
        ({"provider": "openai", "model": "gpt-5"}, ""),
        ({"provider": "openai", "model": "gpt-5", "secretId": "   "}, ""),
        ({"provider": "openai", "model": "gpt-5", "secretId": 7}, ""),
    ],
)
def test_a_direct_selection_records_which_secret_it_runs_with(stored, expected_secret_id):
    raw = json.dumps({
        "version": SCHEDULE_VERSION,
        "cadence": "daily",
        "timezone": "UTC",
        "paused": False,
        "model": stored,
    })
    schedule = parse_detection_schedule(raw)
    assert schedule.model.secret_id == expected_secret_id
    # Round-trips, so a save does not drop the choice the owner made.
    assert (
        parse_detection_schedule(
            json.dumps({
                "version": SCHEDULE_VERSION,
                "cadence": "daily",
                "timezone": "UTC",
                "paused": False,
                "model": schedule.model.to_dict(),
            })
        ).model.secret_id
        == expected_secret_id
    )


def test_an_endpoint_selection_stores_no_secret_id():
    # An AI Gateway endpoint carries its own credential server-side, so there is nothing to name.
    model = ScheduleModel(endpoint_name="prod-gateway", secret_id="s-ignored")
    assert model.to_dict() == {"endpointName": "prod-gateway"}


def test_a_model_naming_neither_an_endpoint_nor_a_provider_and_model_is_rejected():
    # Every field defaults to "", which would otherwise build the URI ":/".
    with pytest.raises(ValueError, match="names either an AI Gateway endpoint"):
        ScheduleModel()
    with pytest.raises(ValueError, match="names either an AI Gateway endpoint"):
        ScheduleModel(provider="openai")
    assert ScheduleModel(endpoint_name="my-endpoint").to_model_uri() == "gateway:/my-endpoint"
    assert ScheduleModel(provider="OpenAI", model="gpt-5").to_model_uri() == "openai:/gpt-5"
