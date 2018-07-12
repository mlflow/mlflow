from abc import abstractmethod
import os
import subprocess

from mlflow.utils.logging_utils import eprint


def _update_run_status(active_run, status):
    if active_run:
        active_run.set_terminated(status)


class PollableRun(object):
    def __init__(self):
        pass

    @abstractmethod
    def wait_impl(self):
        pass

    @abstractmethod
    def cancel(self):
        """
        Cancels the pollable run (interrupts the command subprocess, cancels the
        Databricks run, etc)
        """
        pass

    def pre_monitor_hook(self):
        """
        Hook to execute before the monitoring loop in our subprocess. Can be used e.g. to launch
        a local run in a subprocess.
        """
        pass

    def monitor_run(self, active_run):
        """
        Polls the run for termination, sending updates on the run's status to a tracking server via
        the passed-in `ActiveRun` instance.
        """
        self.pre_monitor_hook()
        run_id = active_run.get_run().info.run_uuid if active_run else "unknown"
        try:
            run_succeeded = self.wait_impl()
            if run_succeeded:
                eprint("=== Run (ID '%s') succeeded ===" % run_id)
                _update_run_status(active_run, "FINISHED")
            else:
                eprint("=== Run (ID '%s') failed ===" % run_id)
                _update_run_status(active_run, "FAILED")
        except KeyboardInterrupt:
            eprint("=== Run was (ID '%s') interrupted, cancelling run... ===" % run_id)
            _update_run_status(active_run, "FAILED")
        finally:
            self.cancel()


def _launch_command(command, work_dir, env_map, stream_output):
    """
    Launch entry point command in a subprocess, returning a `subprocess.Popen` representing the
    subprocess. The subprocess's stderr & stdout are streamed to the current process's
    stderr & stdout.
    """
    cmd_env = os.environ.copy()
    cmd_env.update(env_map)
    if stream_output:
        return subprocess.Popen([os.environ.get("SHELL", "bash"), "-c", command],
                                cwd=work_dir, env=cmd_env)
    # TODO: Can this hang if there's too much stdout/stderr buffered?
    return subprocess.Popen(
        [os.environ.get("SHELL", "bash"), "-c", command],
        cwd=work_dir, env=cmd_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


class LocalPollableRun(PollableRun):
    def __init__(self, command, work_dir, env_map, stream_output):
        super(LocalPollableRun, self).__init__()
        self.command = command
        self.work_dir = work_dir
        self.env_map = env_map
        self.stream_output = stream_output

    def pre_monitor_hook(self):
        self.command_proc = _launch_command(
            self.command, self.work_dir, self.env_map, self.stream_output)

    def wait_impl(self):
        return self.command_proc.wait() == 0

    def cancel(self):
        try:
            self.command_proc.terminate()
        except OSError:
            pass


class DatabricksPollableRun(PollableRun):
    def __init__(self, databricks_run_id):
        super(DatabricksPollableRun, self).__init__()
        self.databricks_run_id = databricks_run_id

    def wait_impl(self):
        from mlflow.projects import databricks
        return databricks.monitor_databricks(self.databricks_run_id)

    def cancel(self):
        from mlflow.projects import databricks
        databricks._jobs_runs_cancel(self.databricks_run_id)
