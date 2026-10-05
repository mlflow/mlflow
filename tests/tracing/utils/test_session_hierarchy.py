import json

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracing.utils.session_hierarchy import (
    parse_session_hierarchy_levels,
    parse_session_path,
    session_group_key,
    set_session_hierarchy,
)
from mlflow.tracking.client import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_EXPERIMENT_SESSION_HIERARCHY


def _hierarchy_tag(levels):
    return json.dumps(
        {"version": 1, "levels": [{"name": level} for level in levels]},
        separators=(",", ":"),
    )


# ==================== Tests for parse_session_hierarchy_levels ====================


@pytest.mark.parametrize(
    "tag_value",
    [
        None,
        "",
        "not json",
        '"a string"',
        "[]",
        '{"version": 1}',  # missing levels
        '{"levels": [{"name": "trip"}]}',  # missing version
        '{"version": "1", "levels": [{"name": "trip"}]}',  # version not an int
        '{"version": 1, "levels": []}',  # empty levels
        '{"version": 1, "levels": [{"name": ""}]}',  # empty name
        '{"version": 1, "levels": [{"name": 1}]}',  # non-string name
        '{"version": 1, "levels": [{"name": "trip"}, {"name": "trip"}]}',  # duplicate names
        '{"version": 1, "levels": ["trip"]}',  # level not an object
    ],
)
def test_parse_session_hierarchy_levels_invalid(tag_value):
    assert parse_session_hierarchy_levels(tag_value) is None


def test_parse_session_hierarchy_levels_valid():
    tag_value = _hierarchy_tag(["trip", "episode", "domain"])
    assert parse_session_hierarchy_levels(tag_value) == ["trip", "episode", "domain"]


def test_parse_session_hierarchy_levels_ignores_extra_fields():
    tag_value = json.dumps({"version": 1, "levels": [{"name": "trip", "extra": 1}], "other": True})
    assert parse_session_hierarchy_levels(tag_value) == ["trip"]


# ==================== Tests for parse_session_path ====================


def test_parse_session_path_valid():
    assert parse_session_path('["trip-2","ep-2","dom-2","turn-5"]') == [
        "trip-2",
        "ep-2",
        "dom-2",
        "turn-5",
    ]


@pytest.mark.parametrize(
    "value",
    [
        None,
        123,
        ["trip-2"],  # a raw list is not the serialized (string) form
        "plain-session",
        "[]",  # empty path
        '["trip-2", 1]',  # non-string element
        '"just a string"',  # valid JSON but not an array
        '{"a": 1}',  # valid JSON but not an array
    ],
)
def test_parse_session_path_invalid(value):
    assert parse_session_path(value) is None


# ==================== Tests for session_group_key ====================


def test_session_group_key_exact_level():
    # Path length exactly K=2: the whole path is the key.
    assert session_group_key('["trip-2","ep-2"]', 1) == '["trip-2","ep-2"]'


def test_session_group_key_deeper_path_truncates():
    assert session_group_key('["trip-2","ep-2","dom-2","turn-5"]', 1) == '["trip-2","ep-2"]'
    assert session_group_key('["trip-2","ep-2","dom-2","turn-5"]', 0) == '["trip-2"]'


def test_session_group_key_shorter_path_returns_original():
    assert session_group_key('["trip-2"]', 1) == '["trip-2"]'
    assert session_group_key('["trip-2","ep-2"]', 2) == '["trip-2","ep-2"]'


def test_session_group_key_plain_string_returns_original():
    assert session_group_key("session-1", 0) == "session-1"
    assert session_group_key("session-1", 2) == "session-1"


def test_session_group_key_serializes_compact_json():
    # The key must round-trip through serialize_session_id's compact form so keys
    # built from the same path are equal regardless of the input's spacing.
    spaced = session_group_key('["trip-2", "ep-2"]', 1)
    assert spaced == json.dumps(["trip-2", "ep-2"], separators=(",", ":"))


# ==================== Tests for set_session_hierarchy ====================


@pytest.mark.parametrize(
    "levels",
    [
        [],
        ["", "trip"],
        ["trip", "trip"],
        "trip",
        [1, 2],
    ],
)
def test_set_session_hierarchy_rejects_invalid_levels(levels):
    with pytest.raises(MlflowException, match="`levels` must be a non-empty list"):
        set_session_hierarchy(levels)


def test_set_session_hierarchy_writes_valid_tag():
    experiment_id = mlflow.set_experiment("session-hierarchy-test").experiment_id

    set_session_hierarchy(["trip", "episode"], experiment_id=experiment_id)

    tags = MlflowClient().get_experiment(experiment_id).tags
    tag_value = tags[MLFLOW_EXPERIMENT_SESSION_HIERARCHY]
    assert parse_session_hierarchy_levels(tag_value) == ["trip", "episode"]
    # Compact JSON, versioned schema.
    assert tag_value == _hierarchy_tag(["trip", "episode"])


def test_set_session_hierarchy_defaults_to_current_experiment():
    experiment_id = mlflow.set_experiment("session-hierarchy-default-exp").experiment_id

    set_session_hierarchy(["trip"])

    assert parse_session_hierarchy_levels(
        MlflowClient().get_experiment(experiment_id).tags[MLFLOW_EXPERIMENT_SESSION_HIERARCHY]
    ) == ["trip"]
