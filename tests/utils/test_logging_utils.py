import subprocess
import sys
import uuid

import pytest


@pytest.mark.parametrize(
    ("log_level", "expected"),
    [
        ("DEBUG", True),
        ("INFO", False),
        ("NOTSET", False),
    ],
)
def test_logging_level(log_level: str, expected: bool) -> None:
    random_str = str(uuid.uuid4())
    stdout = subprocess.check_output(
        [
            sys.executable,
            "-c",
            f"from mlflow.utils.logging_utils import _debug; _debug({random_str!r})",
        ],
        env={"MLFLOW_LOGGING_LEVEL": log_level},
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert (random_str in stdout) is expected
