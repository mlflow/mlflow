import logging
import tempfile
from pathlib import Path

import mlflow

_logger = logging.getLogger(__name__)


def save_dspy_module_state(program, file_name: str = "model.json"):
    """
    Save the state of states of dspy `Module` to a temporary directory and log it as an artifact.

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
