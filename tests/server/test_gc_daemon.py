from multiprocessing import Process
from unittest.mock import MagicMock

import pytest

import mlflow.server.gc_daemon as gc_module
from mlflow.utils.process import ShellCommandException


class DummyProcess:
    """Stand-in for multiprocessing.Process so tests run inline & instantly."""

    def __init__(self, target, args):
        self.target = target
        self.args = args
        self.daemon = False
        self.started = False

    def start(self):
        self.started = True
        # Run the target synchronously; makes behaviour deterministic.
        self.target(*self.args)

    def join(self, timeout=None):
        pass


@pytest.fixture
def patch_process(monkeypatch):
    """Patch multiprocessing.Process with DummyProcess."""
    monkeypatch.setattr(gc_module, "Process", DummyProcess)
    return


@pytest.fixture(autouse=True)
def patch_exec(monkeypatch):
    """
    Return a MagicMock we can program with side_effects.
    Autouse so we never run the real command.
    """
    mock = MagicMock()
    monkeypatch.setattr(gc_module, "_exec_cmd", mock)
    return mock


def test_process_is_daemon():
    """run_gc_daemon should mark the spawned process as daemon=True."""
    proc = gc_module.run_gc_daemon(
        interval=0, backend_store_uri="db", artifacts_destination="s3://bucket"
    )

    assert isinstance(proc, Process), "run_gc_daemon must return the created Process instance"
    assert proc.daemon is True, "Automatic GC process should run as a daemon"


def test_gc_stops_after_max_consecutive_failures(patch_process, patch_exec):
    """
    When _exec_cmd keeps failing, _gc_process must stop after MAX_CONSECUTIVE_FAILURES
    to avoid an infinite error loop.
    """
    failures_allowed = gc_module.MAX_CONSECUTIVE_FAILURES
    patch_exec.side_effect = ShellCommandException("always fails")

    gc_module.run_gc_daemon(interval=0, backend_store_uri="db", artifacts_destination="s3://bucket")

    assert patch_exec.call_count == failures_allowed, (
        f"_exec_cmd should be attempted exactly {failures_allowed} times before GC disables itself"
    )


def test_gc_retries_reset_on_success(patch_process, patch_exec):
    """
    After a successful _exec_cmd call, the failure counter should reset,
    so GC does not stop prematurely.
    """
    # Simulate a transient failure followed by a success
    # end with consistent failure to ensure the process stops
    patch_exec.side_effect = (
        [ShellCommandException("boom")] * (gc_module.MAX_CONSECUTIVE_FAILURES - 1)
        + [None]
        + [ShellCommandException("boom")] * (gc_module.MAX_CONSECUTIVE_FAILURES)
    )

    try:
        gc_module.run_gc_daemon(
            interval=0, backend_store_uri="db", artifacts_destination="s3://bucket"
        )
    except StopIteration:
        # if the process loop does not stop at max failures,
        # the patched exec will run out of side_effects to raise
        pytest.fail("run_gc_daemon inner loop does not stop after max consecutive failures")

    assert patch_exec.call_count == gc_module.MAX_CONSECUTIVE_FAILURES * 2, (
        "run_gc_daemon should reset the failure counter after a successful _exec_cmd call"
    )
