import subprocess
import sys
import warnings
from collections.abc import Generator
from pathlib import Path

import pytest

from mlflow.store.fs2db import _resolve_mlruns, migrate
from mlflow.tracking import MlflowClient


@pytest.fixture(scope="module")
def clients(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[tuple[MlflowClient, MlflowClient]]:
    tmp = tmp_path_factory.mktemp("fs2db")
    source = tmp / "source"
    target_uri = f"sqlite:///{tmp / 'migrated.db'}"

    subprocess.check_call(
        [
            sys.executable,
            "-I",
            "fs2db/src/generate_synthetic_data.py",
            "--output",
            source,
            "--size",
            "small",
        ]
    )

    migrate(Path(source), target_uri, progress=False)

    mlruns = _resolve_mlruns(Path(source))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*filesystem.*deprecated.*", category=FutureWarning
        )
        src = MlflowClient(tracking_uri=mlruns.as_uri())
        dst = MlflowClient(tracking_uri=target_uri)
        yield src, dst
