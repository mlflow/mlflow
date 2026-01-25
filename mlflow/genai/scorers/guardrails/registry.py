from __future__ import annotations

from mlflow.exceptions import MlflowException

_VALIDATOR_REGISTRY = {
    "ToxicLanguage": "ToxicLanguage",
    "NSFWText": "NSFWText",
    "DetectJailbreak": "DetectJailbreak",
    "DetectPII": "DetectPII",
    "SecretsPresent": "SecretsPresent",
    "GibberishText": "GibberishText",
}


def get_validator_class(validator_name: str):
    if validator_name not in _VALIDATOR_REGISTRY:
        available = ", ".join(sorted(_VALIDATOR_REGISTRY.keys()))
        raise MlflowException.invalid_parameter_value(
            f"Unknown Guardrails AI validator: '{validator_name}'. Available: {available}"
        )

    from guardrails import hub

    class_name = _VALIDATOR_REGISTRY[validator_name]
    return getattr(hub, class_name)
