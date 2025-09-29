import os
import time
from os.path import dirname
from pathlib import Path

import pytest

from mlflow.server.jobs.util import _exec_job_in_subproc, is_process_alive

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="MLflow job execution is not supported on Windows"
)


def sleep_fn(secs: float, tmpdir: str):
    (Path(tmpdir) / "pid").write_text(str(os.getpid()))
    time.sleep(secs)


def test_exec_job_in_subproc_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", dirname(__file__))

    beg_time = time.time()
    result = _exec_job_in_subproc(
        "test_util.sleep_fn",
        {"secs": 10, "tmpdir": str(tmp_path)},
        timeout=3,
        tmpdir=str(tmp_path),
    )
    assert (time.time() - beg_time) < 3.5
    assert result is None
    job_pid = int((tmp_path / "pid").read_text())
    # assert the job subprocess is killed.
    assert not is_process_alive(job_pid)
