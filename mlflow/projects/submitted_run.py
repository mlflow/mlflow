import logging
import os
import signal
from abc import abstractmethod

from mlflow.entities import RunStatus
from mlflow.utils.annotations import developer_stable

_logger = logging.getLogger(__name__)


@developer_stable
class SubmittedRun:
    """
    Wrapper around an MLflow project run (e.g. a subprocess running an entry point
    command or a Databricks job run) and exposing methods for waiting on and cancelling the run.
    This class defines the interface that the MLflow project runner uses to manage the lifecycle
    of runs launched in different environments (e.g. runs launched locally or on Databricks).

    ``SubmittedRun`` is not thread-safe. That is, concurrent calls to wait() / cancel()
    from multiple threads may inadvertently kill resources (e.g. local processes) unrelated to the
    run.

    NOTE:

        Subclasses of ``SubmittedRun`` must expose a ``run_id`` member containing the
        run's MLflow run ID.
    """

    @abstractmethod
    def wait(self):
        """
        Wait for the run to finish, returning True if the run succeeded and false otherwise. Note
        that in some cases (e.g. remote execution on Databricks), we may wait until the remote job
        completes rather than until the MLflow run completes.
        """

    @abstractmethod
    def get_status(self):
        """
        Get status of the run.
        """

    @abstractmethod
    def cancel(self):
        """
        Cancel the run (interrupts the command subprocess, cancels the Databricks run, etc) and
        waits for it to terminate. The MLflow run status may not be set correctly
        upon run cancellation.
        """

    @property
    @abstractmethod
    def run_id(self):
        pass


class LocalSubmittedRun(SubmittedRun):
    """
    Instance of ``SubmittedRun`` corresponding to a subprocess launched to run an entry point
    command locally.
    """

    def __init__(self, run_id, command_proc):
        super().__init__()
        self._run_id = run_id
        self.command_proc = command_proc

    @property
    def run_id(self):
        return self._run_id

    def wait(self):
        return self.command_proc.wait() == 0

    def cancel(self):
        # Interrupt child process if it hasn't already exited
        if self.command_proc.poll() is None:
            # Kill the the process tree rooted at the child if it's the leader of its own process
            # group, otherwise just kill the child
            try:
                if self.command_proc.pid == os.getpgid(self.command_proc.pid):
                    os.killpg(self.command_proc.pid, signal.SIGTERM)
                else:
                    self.command_proc.terminate()
            except OSError:
                # The child process may have exited before we attempted to terminate it, so we
                # ignore OSErrors raised during child process termination
                _logger.info(
                    "Failed to terminate child process (PID %s) corresponding to MLflow "
                    "run with ID %s. The process may have already exited.",
                    self.command_proc.pid,
                    self._run_id,
                )
            self.command_proc.wait()

    def _get_status(self):
        exit_code = self.command_proc.poll()
        if exit_code is None:
            return RunStatus.RUNNING
        if exit_code == 0:
            return RunStatus.FINISHED
        return RunStatus.FAILED

    def get_status(self):
        return RunStatus.to_string(self._get_status())
