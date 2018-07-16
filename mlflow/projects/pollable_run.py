from abc import abstractmethod
import os
import signal
import subprocess
import sys

from mlflow.entities.run_status import RunStatus
from mlflow.utils.logging_utils import eprint


def maybe_set_run_terminated(active_run, status):
    """
    If the passed-in active run is defined and still running (i.e. hasn't already been terminated
    within user code), mark it as terminated with the passed-in status.
    """
    if active_run and not RunStatus.is_terminated(active_run.get_run().info.status):
        active_run.set_terminated(status)


def monitor_run(pollable_run, active_run):
    """
    Polls the run for termination, sending updates on the run's status to a tracking server via
    the passed-in `ActiveRun` instance. This function is intended to be run asynchronously
    in a subprocess.
    """
    # Add a SIGTERM & SIGINT handler to the current process that cancels the run
    def handler(signal_num, stack_frame):  # pylint: disable=unused-argument
        eprint("=== Run (%s) was interrupted, cancelling run... ===" % pollable_run.describe())
        pollable_run.cancel()
        maybe_set_run_terminated(active_run, "FAILED")
        sys.exit(0)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    # Perform any necessary setup for the pollable run, then wait on it to finish
    pollable_run.setup()
    run_succeeded = pollable_run.wait()
    if run_succeeded:
        eprint("=== Run (%s) succeeded ===" % pollable_run.describe())
        maybe_set_run_terminated(active_run, "FINISHED")
    else:
        eprint("=== Run (%s) failed ===" % pollable_run.describe())
        maybe_set_run_terminated(active_run, "FAILED")


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


def _launch_command(command, work_dir, env_map, stream_output):
    """
    Launch entry point command in a subprocess, returning a `subprocess.Popen` representing the
    subprocess.
    """
    cmd_env = os.environ.copy()
    cmd_env.update(env_map)
    if stream_output:
        return subprocess.Popen([os.environ.get("SHELL", "bash"), "-c", command],
                                cwd=work_dir, env=cmd_env, preexec_fn=os.setsid)
    return subprocess.Popen(
        [os.environ.get("SHELL", "bash"), "-c", command],
        cwd=work_dir, env=cmd_env, stdout=open(os.devnull, 'wb'),
        stderr=open(os.devnull, 'wb'), preexec_fn=os.setsid)


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

    def setup(self):
        self.command_proc = _launch_command(
            self.command, self.work_dir, self.env_map, self.stream_output)

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
