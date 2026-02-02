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
    if validator_name not in _SUPPORTED_VALIDATORS:
        available = ", ".join(sorted(_SUPPORTED_VALIDATORS))
        raise MlflowException.invalid_parameter_value(
            f"Unknown Guardrails AI validator: '{validator_name}'. Available: {available}"
        )

    from guardrails import hub

    return getattr(hub, validator_name)
