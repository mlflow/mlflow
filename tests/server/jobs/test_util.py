import importlib
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
        python_env=None,
        timeout=3,
        env_vars=None,
        tmpdir=str(tmp_path),
    )
    assert (time.time() - beg_time) < 3.5
    assert result is None
    job_pid = int((tmp_path / "pid").read_text())
    # assert the job subprocess is killed.
    assert not is_process_alive(job_pid)


def check_python_env_fn(
    expected_python_version: str,
    expected_module_versions: dict[str, str],
):
    from mlflow.utils import PYTHON_VERSION

    assert PYTHON_VERSION == expected_python_version
    for module_name, expected_version in expected_module_versions.items():
        module = importlib.import_module(module_name)
        assert module.__version__ == expected_version


def test_exec_job_in_subproc_with_python_env(monkeypatch, tmp_path):
    from mlflow.utils.environment import _PythonEnv
    from mlflow.version import VERSION as MLFLOW_VERSION

    monkeypatch.setenv("PYTHONPATH", dirname(__file__))

    result = _exec_job_in_subproc(
        "test_util.check_python_env_fn",
        params={
            "expected_python_version": "3.11.9",
            "expected_module_versions": {"openai": "1.108.2", "mlflow": MLFLOW_VERSION},
        },
        python_env=_PythonEnv(
            python="3.11.9",
            dependencies=[
                "openai==1.108.2",
                "pytest<9",
                dirname(dirname(dirname(dirname(__file__)))),  # mlflow repo home
            ],
        ),
        timeout=None,
        env_vars=None,
        tmpdir=str(tmp_path),
    )
    assert result.succeeded


def check_env_fn(expected_env_vars: dict[str, str]):
    for env_name, env_val in expected_env_vars.items():
        assert os.environ[env_name] == env_val


def test_exec_job_in_subproc_with_env_vars(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", dirname(__file__))

    env_vars = {"TEST_ENV1": "ab", "TEST_ENV2": "123"}
    result = _exec_job_in_subproc(
        "test_util.check_env_fn",
        {"expected_env_vars": env_vars},
        python_env=None,
        timeout=3,
        env_vars=env_vars,
        tmpdir=str(tmp_path),
    )
    assert result.succeeded
