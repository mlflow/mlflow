import sys

from mlflow.entities.run_status import RunStatus
from mlflow.utils.logging_utils import eprint

_all_runs = []


def _add_run(run):
    _all_runs.append(run)


old_hook = sys.excepthook


def _kill_active_runs(type, value, traceback):
    """
    Hook that runs when the program exits with an exception - attempts to cancel all ongoing runs.
    Note that the addition of this hook makes the project execution APIs not fork-safe, in that
    a forked process may attempt to cancel the same set of projects. TODO(Sid): I think actually the
    excepthook won't be overridden upon forking.
    """
    old_hook(type, value, traceback)
    for run in _all_runs:
        run.cancel()


sys.excepthook = _kill_active_runs


class SubmittedRun(object):
    """Class exposing information about an MLflow project run submitted for execution."""
    def __init__(self):
        _add_run(self)

    def wait(self):
        pass


class LocalSubmittedRun(SubmittedRun):

    def __init__(self, run_id, command_proc, command):
        super(LocalSubmittedRun, self).__init__()
        self.run_id = run_id
        self.command_proc = command_proc
        self.command = command

    def wait(self):
        return self.command_proc.wait() == 0

    def cancel(self):
        try:
            os.kill(self.command_proc.pid, signal.SIGINT)
        except OSError:
            pass

    def describe(self):
        return "shell command: '%s'" % self.command

    @property
    def run_id(self):
        """Returns the MLflow run ID of the current run"""
        # TODO: we handle the case where the local client can't access the tracking server to
        # support e.g. running projects on Databricks without specifying a tracking server.
        # Should be able to remove this logic once Databricks has a hosted tracking server.
        if self._active_run:
            return self._active_run.run_info.run_uuid
        return None

    def get_status(self):
        """Gets the human-readable status of the MLflow run from the tracking server."""
        if not self._active_run:
            eprint("Can't get MLflow run status; the run's status has not been "
                   "persisted to an accessible tracking server.")
            return None
        return RunStatus.to_string(self._active_run.get_run().info.status)

    def wait(self):
        """
        Waits for the run to complete. Note that in some cases (e.g. remote execution on
        Databricks), we may wait until the remote job completes rather than until the MLflow run
        completes.
        """
        self._pollable_run_obj.wait()

    def cancel(self):
        """
        Attempts to cancel the current run by interrupting the monitoring process; note that this
        will not cancel the run if it has already completed.
        """
        self._pollable_run_obj.cancel()
