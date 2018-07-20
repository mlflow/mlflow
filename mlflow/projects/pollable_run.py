from abc import abstractmethod

import os
import signal


class PollableRun(object):
    """
    Class wrapping a single unit of execution (e.g. a subprocess running an entry point
    command or a Databricks Job run) that can be polled for exit status / cancelled / waited on.
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
    def cancel(self):
        """
        Cancels the pollable run (interrupts the command subprocess, cancels the
        Databricks run, etc). Note that `setup` may not have finished when this method is called.
        """
        pass

    @abstractmethod
    def describe(self):
        """
        Returns a string describing the current run, used when logging information about run
        success or failure.
        """
        pass


class LocalPollableRun(PollableRun):
    """
    Instance of PollableRun corresponding to a subprocess launched to run an entry point command
    locally.
    """
    def __init__(self, command_proc, command):
        super(LocalPollableRun, self).__init__()
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
