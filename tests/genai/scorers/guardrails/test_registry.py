import warnings

import guardrails.hub
import pytest
from guardrails import Validator, register_validator
from guardrails.classes.validation.validation_result import PassResult
from guardrails_ai.lowercase import LowerCase
from guardrails_ai.regex_match import RegexMatch

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.guardrails.registry import (
    _validator_distribution_name,
    _validator_module_name,
    get_validator_class,
)


@register_validator(name="test/legacy_only_validator", data_type="string")
class LegacyOnlyValidator(Validator):
    def validate(self, value, metadata=None):
        return PassResult()


@pytest.mark.parametrize(
    ("validator_name", "module_name", "distribution_name"),
    [
        ("RegexMatch", "regex_match", "guardrails-ai-regex-match"),
        ("DetectPII", "detect_pii", "guardrails-ai-detect-pii"),
        ("NSFWText", "nsfw_text", "guardrails-ai-nsfw-text"),
        ("ToxicLanguage", "toxic_language", "guardrails-ai-toxic-language"),
        ("DetectJailbreak", "detect_jailbreak", "guardrails-ai-detect-jailbreak"),
        ("SecretsPresent", "secrets_present", "guardrails-ai-secrets-present"),
        ("GibberishText", "gibberish_text", "guardrails-ai-gibberish-text"),
        ("LowerCase", "lower_case", "guardrails-ai-lower-case"),
    ],
)
def test_validator_package_names(validator_name, module_name, distribution_name):
    assert _validator_module_name(validator_name) == module_name
    assert _validator_distribution_name(validator_name) == distribution_name


@pytest.mark.parametrize(
    ("validator_name", "expected_class"),
    [
        # guardrails_ai.regex_match follows the snake_case convention
        ("RegexMatch", RegexMatch),
        # guardrails_ai.lowercase does not, so this exercises the namespace scan
        ("LowerCase", LowerCase),
    ],
)
def test_get_validator_class_resolves_installed_package(validator_name, expected_class):
    with warnings.catch_warnings():
        # The guardrails.hub shim warns when it is used. Resolving through the
        # installed packages must not touch it.
        warnings.simplefilter("error", DeprecationWarning)
        result = get_validator_class(validator_name)

    assert result is expected_class


def test_get_validator_class_falls_back_to_guardrails_hub(monkeypatch):
    monkeypatch.setattr(guardrails.hub, "LegacyOnlyValidator", LegacyOnlyValidator, raising=False)

    assert get_validator_class("LegacyOnlyValidator") is LegacyOnlyValidator


def test_get_validator_class_unknown():
    with pytest.raises(MlflowException, match="Unknown Guardrails AI validator") as exc_info:
        get_validator_class("NoSuchValidator")

    message = str(exc_info.value)
    assert "guardrails_ai.no_such_validator" in message
    assert "pip install guardrails-ai-no-such-validator" in message


def test_get_validator_class_surfaces_missing_dependency(tmp_path, monkeypatch):
    # A validator package that is installed but whose own dependency is missing
    # must raise that ModuleNotFoundError, not be reported as an unknown validator.
    package_dir = tmp_path / "guardrails_ai" / "needs_dep"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("import mlflow_test_missing_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError, match="mlflow_test_missing_dependency"):
        get_validator_class("NeedsDep")
