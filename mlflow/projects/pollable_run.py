from abc import abstractmethod
import os
import subprocess

from mlflow.entities.run_status import RunStatus


def maybe_set_run_terminated(active_run, status):
    """
    If the passed-in active run is defined and still running (i.e. hasn't already been terminated
    within user code), mark it as terminated with the passed-in status.
    """
    if active_run and not RunStatus.is_terminated(active_run.get_run().info.status):
        active_run.set_terminated(status)


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

    def setup(self):
        """
        Hook that gets called within `monitor_run` before attempting to wait on the run.
        """
        pass


class LocalPollableRun(PollableRun):
    """
    Instance of PollableRun corresponding to a subprocess launched to run an entry point command
    locally.
    """
    def __init__(self, command, work_dir, env_map, stream_output):
        super(LocalPollableRun, self).__init__()
        self.command = command
        self.work_dir = work_dir
        self.env_map = env_map
        self.stream_output = stream_output
        self.command_proc = None

    def _launch_command(self):
        """
        Launch entry point command in a subprocess, returning a `subprocess.Popen` representing the
        subprocess.
        """
        cmd_env = os.environ.copy()
        cmd_env.update(self.env_map)
        if self.stream_output:
            return subprocess.Popen([os.environ.get("SHELL", "bash"), "-c", self.command],
                                    cwd=self.work_dir, env=cmd_env, preexec_fn=os.setsid)
        return subprocess.Popen(
            [os.environ.get("SHELL", "bash"), "-c", self.command],
            cwd=self.work_dir, env=cmd_env, stdout=open(os.devnull, 'wb'),
            stderr=open(os.devnull, 'wb'), preexec_fn=os.setsid)

    def setup(self):
        self.command_proc = self._launch_command()

    def wait(self):
        return self.command_proc.wait() == 0

    def cancel(self):
        try:
            if self.command_proc:
                self.command_proc.terminate()
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

    def wait(self):
        from mlflow.projects import databricks
        return databricks.monitor_databricks(self.databricks_run_id)

    def cancel(self):
        from mlflow.projects import databricks
        databricks.cancel_databricks(self.databricks_run_id)

    def describe(self):
        return "Databricks Job run with id: %s" % self.databricks_run_id
