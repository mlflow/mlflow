import pytest

from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
from mlflow.telemetry.parser import LoggedModelParser, RegisteredModelParser
from mlflow.telemetry.schemas import LoggedModelParams, RegisteredModelParams


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
            LoggedModelParams(
                flavor="custom",
            ),
        ),
        (
            {},
            LoggedModelParams(
                flavor="custom",
            ),
        ),
    ],
)
def test_logged_model_parser(arguments, expected_params):
    assert LoggedModelParser.extract_params(arguments) == expected_params


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
def test_registered_model_parser(arguments, expected_params):
    assert RegisteredModelParser.extract_params(arguments) == expected_params
