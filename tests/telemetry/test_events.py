import pytest

from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
from mlflow.telemetry.events import (
    CreateDatasetEvent,
    CreateExperimentEvent,
    CreateLoggedModelEvent,
    CreateModelVersionEvent,
    CreatePromptEvent,
    CreateRegisteredModelEvent,
    CreateRunEvent,
    EvaluateEvent,
    LogAssessmentEvent,
    MergeRecordsEvent,
    StartTraceEvent,
)


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        (
            {"flavor": "mlflow.pyfunc"},
            {"flavor": "pyfunc"},
        ),
        (
            {"flavor": "sklearn"},
            {"flavor": "sklearn"},
        ),
        (
            {
                "flavor": None,
            },
            None,
        ),
        ({}, None),
    ],
)
def test_logged_model_parse_params(arguments, expected_params):
    assert CreateLoggedModelEvent.name == "create_logged_model"
    assert CreateLoggedModelEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        ({"tags": None}, {"is_prompt": False}),
        ({"tags": {}}, {"is_prompt": False}),
        ({"tags": {IS_PROMPT_TAG_KEY: "true"}}, {"is_prompt": True}),
        ({"tags": {IS_PROMPT_TAG_KEY: "false"}}, {"is_prompt": False}),
        ({}, {"is_prompt": False}),
    ],
)
def test_registered_model_parse_params(arguments, expected_params):
    assert CreateRegisteredModelEvent.name == "create_registered_model"
    assert CreateRegisteredModelEvent.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        ({"tags": None}, {"is_prompt": False}),
        ({"tags": {}}, {"is_prompt": False}),
        ({"tags": {IS_PROMPT_TAG_KEY: "true"}}, {"is_prompt": True}),
        ({"tags": {IS_PROMPT_TAG_KEY: "false"}}, {"is_prompt": False}),
        ({}, {"is_prompt": False}),
    ],
)
def test_create_model_version_parse_params(arguments, expected_params):
    assert CreateModelVersionEvent.name == "create_model_version"
    assert CreateModelVersionEvent.parse(arguments) == expected_params


def test_event_name():
    assert CreatePromptEvent.name == "create_prompt"
    assert CreateLoggedModelEvent.name == "create_logged_model"
    assert CreateRegisteredModelEvent.name == "create_registered_model"
    assert CreateModelVersionEvent.name == "create_model_version"
    assert CreateRunEvent.name == "create_run"
    assert CreateExperimentEvent.name == "create_experiment"
    assert LogAssessmentEvent.name == "log_assessment"
    assert StartTraceEvent.name == "start_trace"
    assert EvaluateEvent.name == "evaluate"
    assert CreateDatasetEvent.name == "create_dataset"
    assert MergeRecordsEvent.name == "merge_records"


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        ({"records": [{"test": "data"}]}, {"record_count": 1, "input_type": "list[dict]"}),
        ({"records": [{"a": 1}, {"b": 2}]}, {"record_count": 2, "input_type": "list[dict]"}),
        ({"records": []}, None),
        ({"records": None}, None),
        ({}, None),
        (None, None),
        ({"records": object()}, None),
    ],
)
def test_merge_records_parse_params(arguments, expected_params):
    assert MergeRecordsEvent.parse(arguments) == expected_params
