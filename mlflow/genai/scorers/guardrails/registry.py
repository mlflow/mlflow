from __future__ import annotations

import importlib
import pkgutil
import re

from mlflow.exceptions import MlflowException

_SUPPORTED_VALIDATORS = [
    "ToxicLanguage",
    "NSFWText",
    "DetectJailbreak",
    "DetectPII",
    "SecretsPresent",
    "GibberishText",
]

# Guardrails AI publishes each validator as a standalone PyPI package named
# ``guardrails-ai-<snake_case_name>`` whose code lives in the ``guardrails_ai``
# namespace package, e.g. ``guardrails-ai-detect-pii`` provides
# ``guardrails_ai.detect_pii.DetectPII``.
_VALIDATOR_NAMESPACE = "guardrails_ai"
_DISTRIBUTION_PREFIX = "guardrails-ai-"

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _validator_module_name(validator_name: str) -> str:
    """
    Convert a validator class name to the snake_case module name used by its PyPI package.

    Args:
        validator_name: Validator class name, e.g. ``"DetectPII"`` or ``"NSFWText"``.

    Returns:
        The module name under the ``guardrails_ai`` namespace, e.g. ``"detect_pii"``.
    """
    return _CAMEL_BOUNDARY.sub("_", validator_name).lower()


def _validator_distribution_name(validator_name: str) -> str:
    """
    Return the PyPI distribution that provides a validator, e.g. ``"guardrails-ai-detect-pii"``.
    """
    return _DISTRIBUTION_PREFIX + _validator_module_name(validator_name).replace("_", "-")


def _import_from_namespace(validator_name: str):
    """
    Import a validator class from the installed ``guardrails_ai`` packages.

    Tries ``guardrails_ai.<snake_case_name>`` first. A few packages do not follow
    that convention (``guardrails-ai-lowercase`` exports ``LowerCase`` from
    ``guardrails_ai.lowercase``), so if the direct import misses, every installed
    ``guardrails_ai.*`` package is scanned for an attribute with that name.

    Returns:
        The validator class, or ``None`` when no installed package provides it.

    Raises:
        ModuleNotFoundError: If the package is installed but one of its own
            dependencies is missing, so the real cause is not hidden.
    """
    module_name = f"{_VALIDATOR_NAMESPACE}.{_validator_module_name(validator_name)}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name not in (module_name, _VALIDATOR_NAMESPACE):
            raise
    else:
        if (validator_class := getattr(module, validator_name, None)) is not None:
            return validator_class

    try:
        namespace = importlib.import_module(_VALIDATOR_NAMESPACE)
    except ModuleNotFoundError:
        return None
    for info in pkgutil.iter_modules(namespace.__path__, f"{_VALIDATOR_NAMESPACE}."):
        if info.name == module_name:
            continue
        try:
            module = importlib.import_module(info.name)
        except Exception:
            # An unrelated validator package that fails to import must not block
            # resolution of the one being asked for.
            continue
        if (validator_class := getattr(module, validator_name, None)) is not None:
            return validator_class
    return None


def _import_from_hub(validator_name: str):
    """
    Import a validator class through the deprecated ``guardrails.hub`` shim.

    ``guardrails.hub`` resolves names from the local hub registry and, since
    guardrails-ai 0.11, from the installed ``guardrails_ai`` namespace with a
    deprecation warning. The module is slated for removal upstream, so this is a
    fallback only.

    Returns:
        The validator class, or ``None`` when it cannot be resolved this way.
    """
    try:
        from guardrails import hub
    except ImportError:
        return None
    try:
        return getattr(hub, validator_name)
    except (AttributeError, ImportError):
        return None


def get_validator_class(validator_name: str):
    """
    Get Guardrails AI validator class by name.

    Resolution order:

    1. The validator's own PyPI package, ``guardrails_ai.<snake_case_name>``, then any
       other installed ``guardrails_ai.*`` package that exports the name.
    2. The deprecated ``guardrails.hub`` module, for validators registered through
       the legacy hub registry.

    Args:
        validator_name: Name of the validator (e.g., "ToxicLanguage", "DetectPII")

    Returns:
        The Guardrails AI validator class

    Raises:
        MlflowException: If the validator cannot be found by either route.
    """
    validator_class = _import_from_namespace(validator_name)
    if validator_class is None:
        validator_class = _import_from_hub(validator_name)
    if validator_class is not None:
        return validator_class

    available = ", ".join(sorted(_SUPPORTED_VALIDATORS))
    # error_code is INVALID_PARAMETER_VALUE but this is an attribute lookup failure
    raise MlflowException.invalid_parameter_value(
        f"Unknown Guardrails AI validator: '{validator_name}'. Could not import "
        f"'{validator_name}' from '{_VALIDATOR_NAMESPACE}."
        f"{_validator_module_name(validator_name)}' or from 'guardrails.hub'. "
        f"Guardrails AI validators are published to PyPI as separate packages, usually "
        f"`pip install {_validator_distribution_name(validator_name)}`. Install it and retry. "
        f"Pre-configured scorers: {available}",
        error_class="ATTRIBUTE_NOT_FOUND",
    )
