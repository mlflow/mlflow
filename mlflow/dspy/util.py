import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from dspy import Example

import mlflow

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
    except Exception as e:
        _logger.warning(f"Failed to save dspy module state: {e}")


def log_dspy_module_params(program):
    """
    Log the parameters of the dspy `Module` as run parameters.

    Args:
        program: The dspy `Module` to be logged.
    """
    try:
        states = program.dump_state()
        flat_state_dict = _flatten_dspy_module_state(
            states, exclude_keys=("metadata", "lm", "traces", "train")
        )
        mlflow.log_params(
            {f"{program.__class__.__name__}.{k}": v for k, v in flat_state_dict.items()}
        )
    except Exception as e:
        _logger.warning(f"Failed to log dspy module params: {e}")


def log_dspy_dataset(dataset: list["Example"], file_name: str):
    """
    Log the DSPy dataset as a table.

    Args:
        dataset: The dataset to be logged.
        file_name: The name of the file to save the dataset.
    """
    result = defaultdict(list)
    try:
        for example in dataset:
            for k, v in example.items():
                result[k].append(v)
        mlflow.log_table(result, file_name)
    except Exception as e:
        _logger.warning(f"Failed to log dataset: {e}")


def _flatten_dspy_module_state(
    d, parent_key="", sep=".", exclude_keys: set[str] | None = None
) -> dict[str, Any]:
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
        >>> _flatten_dspy_module_state({"a": {"b": [5, 6]}})
        {'a.b.0': 5, 'a.b.1': 6}
    """
    items: dict[str, Any] = {}

    if isinstance(d, dict):
        for k, v in d.items():
            if exclude_keys and k in exclude_keys:
                continue
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, Example):
                # Don't flatten Example objects further even if it has dict or list values
                v = {key: str(value) for key, value in v.items()}
            items.update(_flatten_dspy_module_state(v, new_key, sep))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if isinstance(v, Example):
                # Don't flatten Example objects further even if it has dict or list values
                v = {key: str(value) for key, value in v.items()}
            items.update(_flatten_dspy_module_state(v, new_key, sep))
    else:
        if d is not None:
            items[parent_key] = d

    return items
