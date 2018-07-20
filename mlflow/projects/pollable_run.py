from abc import abstractmethod

import os
import signal


class PollableRunStatus(object):
    RUNNING, SCHEDULED, FINISHED, FAILED = range(1, 5)
    _STRING_TO_STATUS = {
        "RUNNING": RUNNING,
        "SCHEDULED": SCHEDULED,
        "FINISHED": FINISHED,
        "FAILED": FAILED,
    }
    _STATUS_TO_STRING = {value: key for key, value in _STRING_TO_STATUS.items()}
    _TERMINATED_STATUSES = set([FINISHED, FAILED])

    @staticmethod
    def to_string(status):
        if status not in PollableRunStatus._STATUS_TO_STRING:
            raise Exception("Could not get string corresponding to run status %s. Valid run "
                            "statuses: %s" % (status, list(PollableRunStatus._STATUS_TO_STRING.keys())))
        return PollableRunStatus._STATUS_TO_STRING[status]

    @staticmethod
    def from_string(status_str):
        if status_str not in PollableRunStatus._STRING_TO_STATUS:
            raise Exception(
                "Could not get run status corresponding to string %s. Valid run "
                "status strings: %s" % (status_str, list(PollableRunStatus._STRING_TO_STATUS.keys())))
        return PollableRunStatus._STRING_TO_STATUS[status_str]


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

    @abstractmethod
    def get_status(self):
        """
        Returns the run's status as a string.
        :return one of mlflow.projects.pollable_run.{RUNNING, FAILED, FINISHED}
        """


class LocalPollableRun(PollableRun):
    """
    Instance of PollableRun corresponding to a subprocess launched to run an entry point command
    locally.
    """
    def __init__(self, command_proc, command):
        super(LocalPollableRun, self).__init__()
        self.command_proc = command_proc
        self.command = command
        self.status = PollableRunStatus.RUNNING

    def get_status(self):
        return PollableRunStatus.to_string(self.status)

    def wait(self):
        if self.status == PollableRunStatus.RUNNING:
            self.status = PollableRunStatus.FINISHED if (self.command_proc.wait() == 0) else PollableRunStatus.FAILED

    def cancel(self):
        try:
            os.kill(self.command_proc.pid, signal.SIGINT)
        except OSError:
            pass

    def describe(self):
        return "shell command: '%s'" % self.command


class DatabricksPollableRun(PollableRun):
    """
    Instance of PollableRun corresponding to a Databricks Job run launched to run an MLflow project.
    """
    def __init__(self, databricks_run_id):
        super(DatabricksPollableRun, self).__init__()
        self.databricks_run_id = databricks_run_id
        self.status = PollableRunStatus.RUNNING

    def get_status(self):
        return PollableRunStatus.to_string(self.status)

    def wait(self):
        if self.status == PollableRunStatus.RUNNING:
            from mlflow.projects.databricks import monitor_databricks
            self.status = PollableRunStatus.FINISHED if monitor_databricks(self.databricks_run_id) else PollableRunStatus.FAILED

    def cancel(self):
        from mlflow.projects.databricks import cancel_databricks
        cancel_databricks(self.databricks_run_id)

    def describe(self):
        return "Databricks Job run with id: %s" % self.databricks_run_id
