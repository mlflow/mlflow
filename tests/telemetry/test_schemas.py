import pytest

from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
from mlflow.telemetry.schemas import (
    CreateModelVersionParams,
    LoggedModelParams,
    RegisteredModelParams,
)


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        (
            {"flavor": "mlflow.pyfunc"},
            LoggedModelParams(
                flavor="pyfunc",
            ),
        ),
        (
            {"flavor": "sklearn"},
            LoggedModelParams(
                flavor="sklearn",
            ),
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
    assert LoggedModelParams.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        ({"tags": None}, RegisteredModelParams(is_prompt=False)),
        ({"tags": {}}, RegisteredModelParams(is_prompt=False)),
        ({"tags": {IS_PROMPT_TAG_KEY: "true"}}, RegisteredModelParams(is_prompt=True)),
        ({"tags": {IS_PROMPT_TAG_KEY: "false"}}, RegisteredModelParams(is_prompt=False)),
        ({}, RegisteredModelParams(is_prompt=False)),
    ],
)
def test_registered_model_parse_params(arguments, expected_params):
    assert RegisteredModelParams.parse(arguments) == expected_params


@pytest.mark.parametrize(
    ("arguments", "expected_params"),
    [
        ({"tags": None}, CreateModelVersionParams(is_prompt=False)),
        ({"tags": {}}, CreateModelVersionParams(is_prompt=False)),
        ({"tags": {IS_PROMPT_TAG_KEY: "true"}}, CreateModelVersionParams(is_prompt=True)),
        ({"tags": {IS_PROMPT_TAG_KEY: "false"}}, CreateModelVersionParams(is_prompt=False)),
        ({}, CreateModelVersionParams(is_prompt=False)),
    ],
)
def test_create_model_version_parse_params(arguments, expected_params):
    assert CreateModelVersionParams.parse(arguments) == expected_params
