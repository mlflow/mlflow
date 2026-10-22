import json

import pytest

from mlflow.entities.experiment import Experiment
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.genai.discovery.constants import DEFAULT_CATEGORIES
from mlflow.genai.discovery.settings import (
    DetectionSettings,
    parse_detection_settings,
    read_detection_settings,
)
from mlflow.utils.mlflow_tags import MLFLOW_ISSUE_DETECTION_PRIORITIES

ALL_CATEGORIES = tuple(DEFAULT_CATEGORIES)


def _experiment(tags: dict[str, str]) -> Experiment:
    return Experiment(
        experiment_id="1",
        name="exp",
        artifact_location="loc",
        lifecycle_stage="active",
        tags=[ExperimentTag(key, value) for key, value in tags.items()],
    )


@pytest.mark.parametrize(
    "raw_value",
    [None, "", "not json", "[]", '"a string"', "null", json.dumps({"instructions": "focus here"})],
)
def test_anything_unreadable_means_every_category(raw_value: str | None):
    assert parse_detection_settings(raw_value) == DetectionSettings(categories=ALL_CATEGORIES)


def test_a_deselected_category_is_dropped():
    settings = parse_detection_settings(json.dumps({"categories": {"cost": False}}))
    assert "cost" not in settings.categories
    assert set(settings.categories) == set(ALL_CATEGORIES) - {"cost"}


def test_a_built_in_the_tag_does_not_mention_stays_selected():
    # A category added after the owner last saved is not a decision they made.
    settings = parse_detection_settings(json.dumps({"categories": {"correctness": True}}))
    assert settings.categories == ALL_CATEGORIES


def test_deselecting_everything_still_means_every_category():
    # Matches what the tab runs when nothing is ticked.
    all_off = dict.fromkeys(DEFAULT_CATEGORIES, False)
    settings = parse_detection_settings(json.dumps({"categories": all_off}))
    assert settings.categories == ALL_CATEGORIES


def test_a_custom_category_is_kept():
    settings = parse_detection_settings(
        json.dumps({"categories": {"  checkout flow  ": True, "cost": False}})
    )
    assert "checkout flow" in settings.categories
    assert "cost" not in settings.categories


@pytest.mark.parametrize("selected", ["true", 1, None, [], {}])
def test_a_non_boolean_selection_leaves_the_category_as_it_was(selected: object):
    settings = parse_detection_settings(json.dumps({"categories": {"cost": selected}}))
    assert "cost" in settings.categories


def test_a_blank_category_name_is_ignored():
    settings = parse_detection_settings(json.dumps({"categories": {"   ": True, "cost": False}}))
    assert settings.categories == tuple(c for c in ALL_CATEGORIES if c != "cost")


def test_settings_are_read_from_the_experiment_tag():
    experiment = _experiment({
        MLFLOW_ISSUE_DETECTION_PRIORITIES: json.dumps({"categories": {"safety": False}})
    })
    assert "safety" not in read_detection_settings(experiment).categories
    assert read_detection_settings(_experiment({})).categories == ALL_CATEGORIES
