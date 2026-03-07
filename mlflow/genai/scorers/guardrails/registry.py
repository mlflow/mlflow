from __future__ import annotations

from mlflow.exceptions import MlflowException

_SUPPORTED_VALIDATORS = [
    "ToxicLanguage",
    "NSFWText",
    "DetectJailbreak",
    "DetectPII",
    "SecretsPresent",
    "GibberishText",
]


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
