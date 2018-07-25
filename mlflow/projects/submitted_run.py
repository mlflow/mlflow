from abc import abstractmethod

import os
import signal
import sys

from mlflow.entities.run_status import RunStatus
from mlflow.utils.logging_utils import eprint

_all_runs = []


def _add_run(run):
    _all_runs.append(run)


old_hook = sys.excepthook


def _kill_active_runs(exception_type, exception_value, traceback):
    """
    Hook that runs when the program exits with an exception - attempts to cancel all ongoing runs.
    Note that this hook will not run in e.g. IPython notebooks.
    """
    old_hook(exception_type, exception_value, traceback)
    eprint("=== Process %s exiting with exception of type %s, terminating all active project "
           "runs ===" % (os.getpid(), exception_type))
    for run in _all_runs:
        run.cancel()

sys.excepthook = _kill_active_runs


class SubmittedRun(object):
    """
    Class wrapping a MLflow project run (e.g. a subprocess running an entry point
    command or a Databricks Job run) and exposing methods for waiting on / cancelling the run.
    This class defines the interface that the MLflow project runner uses to manage the lifecycle
    of runs launched in different environments (e.g. runs launched locally / on Databricks).

    Note that SubmittedRun is not thread-safe. That is, concurrent calls to wait() / cancel()
    from multiple threads may inadvertently kill resources (e.g. local processes) unrelated to the
    run.
    """
    def __init__(self):
        _add_run(self)

    @abstractmethod
    def wait(self):
        """
        Wait for the run to finish, returning True if the run succeeded and false otherwise. Note
        that in some cases (e.g. remote execution on Databricks), we may wait until the remote job
        completes rather than until the MLflow run completes.
        """
        pass

    @abstractmethod
    def get_status(self):
        """
        Get status of the run.
        """
        pass

    @abstractmethod
    def cancel(self):
        """
        Cancels the run (interrupts the command subprocess, cancels the Databricks run, etc) and
        waits for it to terminate. Note that the MLflow run status may not be set correctly
        upon run cancellation.
        """
        pass

    @abstractmethod
    def describe(self):
        """
        Returns a string describing the current run, used when logging information about run
        success or failure.
        """
        pass


class LocalSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a subprocess launched to run an entry point command
    locally.
    """
    def __init__(self, run_id, command_proc, command):
        super(LocalSubmittedRun, self).__init__()
        self.run_id = run_id
        self.command_proc = command_proc
        self.command = command

    def wait(self):
        return self.command_proc.wait() == 0

    def cancel(self):
        # Interrupt child process if it hasn't already exited
        if self.command_proc.poll() is None:
            # Terminate the child process (MLflow CLI command)
            try:
                self.command_proc.terminate()
            except OSError:
                # The child process may have exited before we attempted to terminate it, so we
                # ignore OSErrors raised during child process termination
                eprint("Failed to terminate child process (PID %s) corresponding to MLflow "
                       "run with ID %s. The process may have already "
                       "exited." % (self.command_proc.pid, self.run_id))
            self.command_proc.wait()
        else:
            eprint("Run %s was not active, unable to cancel." % self.run_id)

    def describe(self):
        return "shell command: '%s'" % self.command

    def _get_status(self):
        exit_code = self.command_proc.poll()
        if exit_code is None:
            return RunStatus.RUNNING
        if exit_code == 0:
            return RunStatus.FINISHED
        return RunStatus.FAILED

    def get_status(self):
        return RunStatus.to_string(self._get_status())
