from abc import abstractmethod

import os
import signal
import sys

from mlflow.entities.run_status import RunStatus

_all_runs = []


def _add_run(run):
    _all_runs.append(run)


old_hook = sys.excepthook


def _kill_active_runs(exception_type, exception_value, traceback):
    """
    Hook that runs when the program exits with an exception - attempts to cancel all ongoing runs.
    Note that the addition of this hook makes the project execution APIs not fork-safe, in that
    a forked process may attempt to cancel the same set of projects. TODO(Sid): I think actually the
    excepthook won't be overridden upon forking.
    """
    old_hook(exception_type, exception_value, traceback)
    print("@SId in excepthook")
    for run in _all_runs:
        run.cancel()


sys.excepthook = _kill_active_runs


class SubmittedRun(object):
    """
    Class wrapping a MLflow project run (e.g. a subprocess running an entry point
    command or a Databricks Job run) and exposing methods for waiting on / cancelling the run.
    This class defines the interface that the MLflow project runner uses to manage the lifecycle
    of runs launched in different environments (e.g. runs launched locally / on Databricks).
    """
    def __init__(self):
        pass

    @abstractmethod
    def wait(self):
        """
        Wait for the run to finish, returning True if the run succeeded and false otherwise.
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
        Cancels the run (interrupts the command subprocess, cancels the Databricks run, etc)
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
        try:
            os.kill(self.command_proc.pid, signal.SIGINT)
        except OSError:
            pass
        self.command_proc.wait()

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
