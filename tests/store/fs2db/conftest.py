import json
import os
import subprocess
import sys
import warnings
from collections.abc import Generator
from pathlib import Path

import pytest

from mlflow.store.fs2db import _resolve_mlruns, migrate
from mlflow.store.tracking.file_store import FileStore
from mlflow.tracing.constant import CostKey, TokenUsageKey, TraceMetadataKey
from mlflow.tracking import MlflowClient


@pytest.fixture(scope="module")
def clients(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch_module: pytest.MonkeyPatch,
) -> Generator[tuple[MlflowClient, MlflowClient]]:
    tmp = tmp_path_factory.mktemp("fs2db")
    source = tmp / "source"
    target_uri = f"sqlite:///{tmp / 'migrated.db'}"

    # Disable async trace logging in the subprocess so traces are written
    # synchronously and immediately available for set_trace_tag calls.
    env = {
        **os.environ,
        "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING": "false",
        "MLFLOW_ALLOW_FILE_STORE": "true",
    }
    subprocess.check_call(
        [
            sys.executable,
            "-I",
            "fs2db/src/generate_synthetic_data.py",
            "--output",
            source,
            "--size",
            "small",
        ],
        env=env,
    )

    mlruns = _resolve_mlruns(Path(source))
    reserved_metadata = {
        TraceMetadataKey.TRACE_SESSION: "session-1",
        TraceMetadataKey.TOKEN_USAGE: json.dumps({
            TokenUsageKey.INPUT_TOKENS: 10,
            TokenUsageKey.OUTPUT_TOKENS: 5,
            TokenUsageKey.TOTAL_TOKENS: 15,
        }),
        TraceMetadataKey.COST: json.dumps({
            CostKey.INPUT_COST: 0.1,
            CostKey.OUTPUT_COST: 0.2,
            CostKey.TOTAL_COST: 0.3,
        }),
    }
    metadata_dirs = mlruns.glob(
        f"*/{FileStore.TRACES_FOLDER_NAME}/*/{FileStore.TRACE_TRACE_METADATA_FOLDER_NAME}"
    )
    for metadata_dir in metadata_dirs:
        for key, value in reserved_metadata.items():
            (metadata_dir / key).write_text(value)

    migrate(Path(source), target_uri, progress=False)

    monkeypatch_module.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="mlflow")
        src = MlflowClient(tracking_uri=mlruns.as_uri())
        dst = MlflowClient(tracking_uri=target_uri)
        yield src, dst
