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
    [None, "", "not json", "[]", '"a string"', "null"],
)
def test_anything_unreadable_means_every_category_and_no_instructions(raw_value: str | None):
    assert parse_detection_settings(raw_value) == DetectionSettings(categories=ALL_CATEGORIES)


def test_saved_instructions_are_read_as_written():
    saved = dict.fromkeys(DEFAULT_CATEGORIES, True) | {"cost": False}
    settings = parse_detection_settings(
        json.dumps({"instructions": "  watch the checkout flow  ", "categories": saved})
    )
    assert settings.instructions == "  watch the checkout flow  "
    assert "cost" not in settings.categories


def test_instructions_survive_a_category_selection_this_server_cannot_read():
    # The categories fall back to every one, but the owner's priorities are not lost with them.
    settings = parse_detection_settings(
        json.dumps({"instructions": "watch the checkout flow", "categories": "all of them"})
    )
    assert settings == DetectionSettings(
        categories=ALL_CATEGORIES, instructions="watch the checkout flow"
    )


@pytest.mark.parametrize("instructions", [None, 7, [], {}, True])
def test_non_text_instructions_read_as_none_saved(instructions: object):
    assert parse_detection_settings(json.dumps({"instructions": instructions})).instructions == ""


def test_a_deselected_category_is_dropped():
    # The shape the tab writes: every built-in, with the unticked ones false.
    saved = dict.fromkeys(DEFAULT_CATEGORIES, True) | {"cost": False}
    settings = parse_detection_settings(json.dumps({"categories": saved}))
    assert settings.categories == tuple(c for c in ALL_CATEGORIES if c != "cost")


def test_only_the_categories_the_tag_selects_are_used():
    # Read as the tab reads it: a built-in the tag does not mention is not selected. A category
    # added to the product after the owner last saved is therefore off until they save again.
    settings = parse_detection_settings(json.dumps({"categories": {"correctness": True}}))
    assert settings.categories == ("correctness",)


def test_deselecting_everything_still_means_every_category():
    # Matches what the tab runs when nothing is ticked.
    all_off = dict.fromkeys(DEFAULT_CATEGORIES, False)
    settings = parse_detection_settings(json.dumps({"categories": all_off}))
    assert settings.categories == ALL_CATEGORIES


def test_a_custom_category_is_kept():
    settings = parse_detection_settings(
        json.dumps({"categories": {"  checkout flow  ": True, "cost": False}})
    )
    assert settings.categories == ("checkout flow",)


def test_a_mixed_selection_reads_the_same_as_the_tab_reads_it():
    # The same input the tab's own tests pin: one deselected built-in, one selected custom area
    # and one deselected custom area. Only the selected key is used, so this server and the
    # browser hand the pipeline the same list.
    settings = parse_detection_settings(
        json.dumps({
            "categories": {"correctness": False, "Authentication": True, "Formatting": False}
        })
    )
    assert settings.categories == ("Authentication",)


def test_a_repeated_category_is_selected_once():
    settings = parse_detection_settings(
        json.dumps({"categories": {"cost": True, "  cost  ": True}})
    )
    assert settings.categories == ("cost",)


@pytest.mark.parametrize("selected", ["true", 1, None, [], {}])
def test_a_non_boolean_selection_is_not_a_selection(selected: object):
    settings = parse_detection_settings(
        json.dumps({"categories": {"correctness": True, "cost": selected}})
    )
    assert settings.categories == ("correctness",)


def test_a_blank_category_name_is_ignored():
    settings = parse_detection_settings(json.dumps({"categories": {"   ": True, "cost": True}}))
    assert settings.categories == ("cost",)


def test_settings_are_read_from_the_experiment_tag():
    experiment = _experiment({
        MLFLOW_ISSUE_DETECTION_PRIORITIES: json.dumps({
            "instructions": "watch the checkout flow",
            "categories": dict.fromkeys(DEFAULT_CATEGORIES, True) | {"safety": False},
        })
    })
    settings = read_detection_settings(experiment)
    assert "safety" not in settings.categories
    assert settings.instructions == "watch the checkout flow"
    assert read_detection_settings(_experiment({})) == DetectionSettings(categories=ALL_CATEGORIES)
