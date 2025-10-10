import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.optimizers.utils import parse_model_name


@pytest.mark.parametrize(
    ("input_name", "expected_output"),
    [
        # URI format (provider:/model) -> provider/model
        ("openai:/gpt-4o", "openai/gpt-4o"),
        ("google:/gemini-pro", "google/gemini-pro"),
        # Standard format (provider/model) -> unchanged
        ("openai/gpt-4o", "openai/gpt-4o"),
        ("google/gemini-pro", "google/gemini-pro"),
    ],
)
def test_parse_model_name_valid_formats(input_name, expected_output):
    assert parse_model_name(input_name) == expected_output

@pytest.mark.parametrize(
    "invalid_name",
    [
        "",
        "   ",
        "gpt-4o",
        "openai:",
        "openai:/",
        "openai//gpt-4o",
        "openai/gpt-4o/extra",
        "openai:gpt-4o",
        "://gpt-4o",
        "openai/",
        "azure_openai:/gpt-4",
    ],
)
def test_parse_model_name_invalid_formats(invalid_name):
    with pytest.raises(
        MlflowException, match="Invalid model name format|cannot be empty"
    ):
        parse_model_name(invalid_name)
