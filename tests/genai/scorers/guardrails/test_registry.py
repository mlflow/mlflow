from unittest.mock import patch

import pytest

from mlflow.exceptions import MlflowException


@pytest.mark.parametrize(
    ("validator_name", "expected_class"),
    [
        ("ToxicLanguage", "ToxicLanguage"),
        ("NSFWText", "NSFWText"),
        ("DetectJailbreak", "DetectPromptInjection"),
        ("DetectPII", "DetectPII"),
        ("SecretsPresent", "SecretsPresent"),
        ("GibberishText", "GibberishText"),
    ],
)
def test_get_validator_class(validator_name, expected_class):
    with (
        patch("mlflow.genai.scorers.guardrails.registry.check_guardrails_installed"),
        patch("guardrails.hub") as mock_hub,
    ):
        mock_hub.configure_mock(**{expected_class: f"{expected_class}Class"})

        from mlflow.genai.scorers.guardrails.registry import get_validator_class

        result = get_validator_class(validator_name)

    assert result == f"{expected_class}Class"


def test_get_validator_class_unknown():
    with patch("mlflow.genai.scorers.guardrails.registry.check_guardrails_installed"):
        from mlflow.genai.scorers.guardrails.registry import get_validator_class

        with pytest.raises(MlflowException, match="Unknown Guardrails AI validator"):
            get_validator_class("UnknownValidator")


def test_list_available_validators():
    from mlflow.genai.scorers.guardrails.registry import list_available_validators

    validators = list_available_validators()

    assert "ToxicLanguage" in validators
    assert "DetectPII" in validators
    assert "DetectJailbreak" in validators
    assert "GibberishText" in validators
    assert len(validators) == 6
