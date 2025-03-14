import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from dspy import Example

_logger = logging.getLogger(__name__)


def save_dspy_module_state(program, file_name: str = "model.json"):
    """
    Save states of dspy `Module` to a temporary directory and log it as an artifact.

    Args:
        program: The dspy `Module` to be saved.
        file_name: The name of the file to save the dspy module state. Default is `model.json`.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, file_name)
            program.save(path)
            mlflow.log_artifact(path)
    except Exception:
        _logger.warning("Failed to save dspy module state", exc_info=True)


def log_dspy_module_state_params(program):
    try:
        states = program.dump_state()
        mlflow.log_params(_flatten_dict(states, exclude_keys=("metadata", "lm", "traces", "train")))
    except Exception:
        _logger.warning("Failed to log dspy module params", exc_info=True)


def log_dataset(dataset: list["Example"], file_name: str):
    result = defaultdict(list)
    try:
        for example in dataset:
            for k, v in example.items():
                result[k].append(v)
        mlflow.log_table(result, file_name)
    except Exception:
        _logger.warning("Failed to log dataset", exc_info=True)


def _flatten_dict(d, parent_key="", sep=".", exclude_keys=()) -> dict:
    """
    Flattens a nested dictionary and accumulates the key names.

    Args:
        d: The dictionary or list to flatten.
        parent_key: The base key used in recursion. Defaults to "".
        sep: Separator for nested keys. Defaults to '.'.
        exclude_keys: Keys to exclude from the flattened dictionary. Defaults to ().

    Returns:
        dict: A flattened dictionary with accumulated keys.

    Example:
        >>> _flatten_dict({"a": {"b": [5, 6]}})
        {'a.b.0': 5, 'a.b.1': 6}
    """
    items = {}

    if isinstance(d, dict):
        for k, v in d.items():
            if k in exclude_keys:
                continue
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if hasattr(v, "toDict"):
                v = v.toDict()
            items.update(_flatten_dict(v, new_key, sep))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if hasattr(v, "toDict"):
                v = v.toDict()
            items.update(_flatten_dict(v, new_key, sep))
    else:
        if d is not None:
            items[parent_key] = d

    return items
