import os
import signal
import subprocess
import sys
import tempfile

from mlflow.utils.logging_utils import eprint
from mlflow.projects.utils import _load_project, _get_conda_env_name, _maybe_set_run_terminated
from mlflow import tracking


def _get_storage_dir(storage_dir):
    if storage_dir is not None and not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    return tempfile.mkdtemp(dir=storage_dir)


def _run_and_monitor_entry_point(
        uri, entry_point, use_conda, parameters, storage_dir):
    project = _load_project(work_dir=uri, uri=uri)
    storage_dir_for_run = _get_storage_dir(storage_dir)
    eprint("=== Created directory %s for downloading remote URIs passed to arguments of "
           "type 'path' ===" % storage_dir_for_run)
    commands = []
    if use_conda:
        # Assume conda env was already created
        conda_env_path = os.path.abspath(os.path.join(uri, project.conda_env))
        commands.append("source activate %s" % _get_conda_env_name(conda_env_path))
    commands.append(
        project.get_entry_point(entry_point).compute_command(parameters, storage_dir_for_run))

    command = " && ".join(commands)
    run_id = os.environ[tracking._RUN_ID_ENV_VAR]
    store = tracking._get_store()
    run_info = tracking.get_run(run_id).info
    active_run = tracking.ActiveRun(store=store, run_info=run_info)
    eprint("=== Running command '%s' in run with ID '%s' === " % (command, run_id))
    # Set up signal handler to terminate the subprocess running the entry-point command
    process = None

    def handle_cancellation(signum, frame):  # pylint: disable=unused-argument
        eprint("=== Shell command '%s' interrupted, cancelling... ===" % command)
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        # Mark the run as terminated if it hasn't already finished
        _maybe_set_run_terminated(active_run, "FAILED")
        sys.exit(1)
    signal.signal(signal.SIGTERM, handle_cancellation)
    process = subprocess.Popen(["bash", "-c", command], close_fds=True, preexec_fn=os.setsid)
    exit_code = process.wait()
    if exit_code == 0:
        eprint("=== Shell command '%s' succeeded ===" % command)
        _maybe_set_run_terminated(active_run, "FINISHED")
        sys.exit(exit_code)
    else:
        eprint("=== Shell command '%s' failed with exit code %s ===" % (command, exit_code))
        _maybe_set_run_terminated(active_run, "FAILED")
        sys.exit(exit_code)