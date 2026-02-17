from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("guardrails")

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.guardrails.registry import get_validator_class


@pytest.mark.parametrize(
    ("validator_name", "expected_class"),
    [
        ("ToxicLanguage", "ToxicLanguage"),
        ("NSFWText", "NSFWText"),
        ("DetectJailbreak", "DetectJailbreak"),
        ("DetectPII", "DetectPII"),
        ("SecretsPresent", "SecretsPresent"),
        ("GibberishText", "GibberishText"),
    ],
)
def test_get_validator_class(validator_name, expected_class):
    with patch("guardrails.hub") as mock_hub:
        mock_hub.configure_mock(**{expected_class: f"{expected_class}Class"})

        result = get_validator_class(validator_name)

    assert result == f"{expected_class}Class"


def test_get_validator_class_unknown():
    with patch("guardrails.hub", spec=[]):
        with pytest.raises(MlflowException, match="Unknown Guardrails AI validator"):
            get_validator_class("UnknownValidator")


def test_get_validator_class_dynamic_import():
    with patch("guardrails.hub") as mock_hub:
        mock_validator = MagicMock()
        mock_hub.NewValidator = mock_validator

        result = get_validator_class("NewValidator")

        assert result is mock_validator
