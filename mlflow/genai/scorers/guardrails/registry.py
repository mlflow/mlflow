from __future__ import annotations

import importlib

from mlflow.exceptions import MlflowException

_SUPPORTED_VALIDATORS = [
    "ToxicLanguage",
    "NSFWText",
    "DetectJailbreak",
    "DetectPII",
    "SecretsPresent",
    "GibberishText",
    "RegexMatch",
]

# Some hub validators trigger a circular import when loaded via guardrails.hub.__init__
# (hub.__init__ eagerly imports all installed validators, which can re-enter guardrails
# before it finishes initializing). For these we load the package directly.
_DIRECT_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "RegexMatch": ("guardrails_grhub_regex_match.main", "RegexMatch"),
}


def get_validator_class(validator_name: str):
    """
    Get Guardrails AI validator class by name.

    For validators in the supported list, imports from guardrails.hub. For unknown
    validators, attempts to dynamically import from guardrails.hub.<ValidatorName>.

    Args:
        validator_name: Name of the validator (e.g., "ToxicLanguage", "DetectPII")

    Returns:
        The Guardrails AI validator class

    Raises:
        MlflowException: If the validator cannot be imported or guardrails is not installed
    """
    if validator_name in _DIRECT_IMPORT_MAP:
        module_path, class_name = _DIRECT_IMPORT_MAP[validator_name]
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise MlflowException.invalid_parameter_value(
                f"Could not import '{validator_name}'. "
                f"Run: guardrails hub install hub://guardrails/regex_match\n{e}"
            )

    from guardrails import hub

    try:
        return getattr(hub, validator_name)
    except AttributeError:
        available = ", ".join(sorted(_SUPPORTED_VALIDATORS))
        raise MlflowException.invalid_parameter_value(
            f"Unknown Guardrails AI validator: '{validator_name}'. Could not find "
            f"'{validator_name}' in 'guardrails.hub'. "
            f"Available pre-configured validators: {available}"
        )
