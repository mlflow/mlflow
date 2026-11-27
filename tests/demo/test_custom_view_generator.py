import json

import pytest

from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DemoFeature, DemoResult
from mlflow.demo.generators.custom_view import (
    DEMO_CUSTOM_VIEW_CREATED_AT_MS,
    DEMO_CUSTOM_VIEW_ID,
    DEMO_CUSTOM_VIEW_INSTRUCTION,
    DEMO_CUSTOM_VIEW_LABEL,
    DEMO_CUSTOM_VIEW_NAME,
    DEMO_CUSTOM_VIEW_TAG_KEY,
    CustomViewDemoGenerator,
    serialize_demo_custom_view,
)
from mlflow.utils.mlflow_tags import MLFLOW_CUSTOM_VIEW_TAG_PREFIX
from mlflow.utils.validation import MAX_EXPERIMENT_TAG_VAL_LENGTH


@pytest.fixture
def custom_view_generator():
    generator = CustomViewDemoGenerator()
    original_version = generator.version
    yield generator
    CustomViewDemoGenerator.version = original_version


def _create_demo_experiment():
    import mlflow

    return mlflow.set_experiment(DEMO_EXPERIMENT_NAME)


def test_generator_attributes():
    generator = CustomViewDemoGenerator()
    assert generator.name == DemoFeature.CUSTOM_VIEW
    assert generator.version == 1
    assert DEMO_CUSTOM_VIEW_TAG_KEY == f"{MLFLOW_CUSTOM_VIEW_TAG_PREFIX}.v1.{DEMO_CUSTOM_VIEW_ID}"


def test_serialized_view_fits_experiment_tag_limit():
    payload = serialize_demo_custom_view()
    assert len(payload) <= MAX_EXPERIMENT_TAG_VAL_LENGTH


def test_serialized_view_shape():
    view = json.loads(serialize_demo_custom_view())
    assert view["id"] == DEMO_CUSTOM_VIEW_ID
    assert view["name"] == DEMO_CUSTOM_VIEW_NAME
    assert view["label"] == DEMO_CUSTOM_VIEW_LABEL
    assert view["instruction"] == DEMO_CUSTOM_VIEW_INSTRUCTION
    assert view["createdAtMs"] == DEMO_CUSTOM_VIEW_CREATED_AT_MS

    template = view["template"]
    assert isinstance(template, list)
    assert template[0]["version"] == "v0.9"
    components = template[0]["updateComponents"]["components"]
    by_id = {component["id"]: component for component in components}

    assert by_id["root"]["component"] == "Column"
    assert by_id["stat-status"]["value"] == {"$source": "metrics.status"}
    assert "assessments" not in by_id
    assert "accuracy" not in by_id

    assert by_id["submit"]["component"] == "FeedbackSubmit"
    assert by_id["submit"]["formId"] == "feedback"
    assert sum(1 for component in components if component.get("component") == "FeedbackSubmit") == 1

    radios = [component for component in components if component.get("component") == "RadioGroup"]
    rationales = [
        component for component in components if component.get("component") == "FeedbackInputText"
    ]
    assert radios
    assert len(radios) == len(rationales)
    assert all(radio["name"] == "Accuracy" for radio in radios)
    assert all(radio["formId"] == "feedback" for radio in radios)
    assert all(radio["spanId"]["$spanRef"] for radio in radios)
    assert [option["value"] for option in radios[0]["options"]] == [
        "Super accurate",
        "Accurate",
        "Somewhat accurate",
        "Not very accurate",
        "Not accurate",
    ]
    assert by_id["root-accuracy"]["spanId"] == {"$spanRef": "root"}
    assert by_id["root-why"]["spanId"] == {"$spanRef": "root"}
    assert by_id["llm0-accuracy"]["spanId"] == {"$spanRef": {"type": "LLM", "nth": 0}}
    assert by_id["llm0-why"]["spanId"] == {"$spanRef": {"type": "LLM", "nth": 0}}

    child_cards = [
        component
        for component in components
        if component.get("component") == "Card" and component["id"] != "root-card"
    ]
    assert child_cards
    assert all("renderIfSpan" in card for card in child_cards)
    assert by_id["root-card"]["renderIfSpan"] == "root"
    assert by_id["llm0-card"]["renderIfSpan"] == {"type": "LLM", "nth": 0}
    assert by_id["tool1-card"]["renderIfSpan"] == {"type": "TOOL", "nth": 1}
    assert "chain0-card" not in by_id
    assert by_id["chain1-card"]["renderIfSpan"] == {"type": "CHAIN", "nth": 1}


def test_data_exists_false_when_no_experiment():
    generator = CustomViewDemoGenerator()
    assert generator._data_exists() is False


def test_generate_requires_demo_experiment():
    generator = CustomViewDemoGenerator()
    with pytest.raises(ValueError, match="not found"):
        generator.generate()


def test_generate_writes_experiment_tag():
    _create_demo_experiment()
    generator = CustomViewDemoGenerator()
    result = generator.generate()

    assert isinstance(result, DemoResult)
    assert result.feature == DemoFeature.CUSTOM_VIEW
    assert result.entity_ids == [DEMO_CUSTOM_VIEW_ID]
    assert "/experiments/" in result.navigation_url

    import mlflow

    experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    stored = json.loads(experiment.tags[DEMO_CUSTOM_VIEW_TAG_KEY])
    assert stored["id"] == DEMO_CUSTOM_VIEW_ID
    assert stored["id"] == DEMO_CUSTOM_VIEW_TAG_KEY.rsplit(".", 1)[-1]


def test_data_exists_true_after_generate():
    _create_demo_experiment()
    generator = CustomViewDemoGenerator()
    assert generator._data_exists() is False

    generator.generate()

    assert generator._data_exists() is True


def test_delete_demo_removes_tag():
    _create_demo_experiment()
    generator = CustomViewDemoGenerator()
    generator.generate()
    assert generator._data_exists() is True

    generator.delete_demo()

    assert generator._data_exists() is False
    import mlflow

    experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    assert DEMO_CUSTOM_VIEW_TAG_KEY not in experiment.tags


def test_is_generated_checks_version(custom_view_generator):
    _create_demo_experiment()
    custom_view_generator.generate()
    custom_view_generator.store_version()

    assert custom_view_generator.is_generated() is True

    CustomViewDemoGenerator.version = 99
    fresh_generator = CustomViewDemoGenerator()
    assert fresh_generator.is_generated() is False
    assert fresh_generator._data_exists() is False
