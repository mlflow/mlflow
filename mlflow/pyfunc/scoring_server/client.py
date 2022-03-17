import requests
import time
from mlflow.pyfunc import scoring_server
import subprocess
import os
import sys
import signal
import json
import atexit
import warnings


class ScoringServerClient:
    def __init__(self, host, port):
        self.url_prefix = f"http://{host}:{port}"

    def ping(self):
        ping_status = requests.get(url=self.url_prefix + "/ping")
        if ping_status.status_code != 200:
            raise Exception(f"ping failed (error code {ping_status.status_code})")

    def wait_server_ready(self, timeout=30):
        begin_time = time.time()
        while True:
            time.sleep(0.3)
            try:
                self.ping()
                return
            except Exception:
                pass
            if time.time() - begin_time > timeout:
                break
        raise RuntimeError("Wait scoring server ready timeout.")

    def get_module_version(self, module_name):
        """
        Get module version on the scoring server worker python environment.
        This method is for testing purpose, i.e., when the server launched in the
        restored python environment, via the client, we can query the python module
        on the running server worker. So that in test code we can confirm the
        server worker are running on correct restored python environment.
        """
        status = requests.get(url=self.url_prefix + f"/version/{module_name}")
        if status.status_code != 200:
            raise Exception(f"get_module_version failed (error code {status.status_code})")
        return status.text.strip()

    def invoke(self, data, pandas_orient="records"):
        import pandas as pd

        content_type_list = []
        if isinstance(data, pd.DataFrame):
            content_type_list.append(scoring_server.CONTENT_TYPE_JSON)
            if pandas_orient == "records":
                content_type_list.append(scoring_server.CONTENT_TYPE_FORMAT_RECORDS_ORIENTED)
            elif pandas_orient == "split":
                content_type_list.append(scoring_server.CONTENT_TYPE_FORMAT_SPLIT_ORIENTED)
            else:
                raise Exception(
                    "Unexpected pandas_orient for Pandas dataframe input %s" % pandas_orient
                )
        else:
            raise RuntimeError("Unsupported data type.")

        post_data = json.dumps(scoring_server._get_jsonable_obj(data, pandas_orient=pandas_orient))

        response = requests.post(
            url=self.url_prefix + "/invocations",
            data=post_data,
            headers={"Content-Type": "; ".join(content_type_list)},
        )

        if response.status_code != 200:
            raise Exception(
                f"Invocation failed (error code {response.status_code}, response: {response.text})"
            )

        return scoring_server.load_predictions_from_json_str(response.text)


def prepare_env(local_model_path, stdout=sys.stdout, stderr=sys.stderr):
    cmd = [
        "mlflow",
        "models",
        "prepare-env",
        "-m",
        local_model_path,
    ]
    if "MLFLOW_HOME" in os.environ:
        cmd.append("--install-mlflow")
    return subprocess.run(cmd, stdout=stdout, stderr=stderr, universal_newlines=True, check=True)


def start_server(
    server_port,
    local_model_path,
    host="127.0.0.1",
    num_workers=1,
    no_conda=False,
    env=None,
    stdout=sys.stdout,
    stderr=sys.stderr,
    sigterm_on_parent_death=True,
):
    cmd = [
        "mlflow",
        "models",
        "serve",
        "-m",
        local_model_path,
        "-h",
        host,
        "-p",
        str(server_port),
        "-w",
        str(num_workers),
    ]
    if no_conda:
        cmd.append("--no-conda")
    elif "MLFLOW_HOME" in os.environ:
        cmd.append("--install-mlflow")

    def sigterm_on_parent_death():
        """
        Uses prctl to automatically send SIGTERM to the command process when its parent is dead.

        This handles the case when the parent is a PySpark worker process.
        If a user cancels the PySpark job, the worker process gets killed, regardless of
        PySpark daemon and worker reuse settings.
        We use prctl to ensure the command process receives SIGTERM after spark job cancellation.
        The command process itself should handle SIGTERM properly.
        This is a no-op on macOS because prctl is not supported.

        Note: we cannot use `atexit` registering "kill_server" handler to replace `prctl` because:
        The functions registered via "atexit" are not called when the program is killed
        by a signal not handled by Python, when a Python fatal internal error is detected, or
        when os._exit().
        """
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            # Set the parent process death signal of the command process to SIGTERM.
            libc.prctl(1,  # PR_SET_PDEATHSIG, see prctl.h
                       signal.SIGTERM)
        except OSError:
            # TODO: find approach for supporting MacOS/Windows system which does not have prctl.
            warnings.warn(
                "System does not support prctl, so when spark job canceled, we have no "
                "way to kill the scoring server if the server launched inside spark UDF.")
            pass

    def preexec_fn():
        # Assign the scoring process to a process group. All child processes of the
        # scoring process will be assigned to this group as well. This allows child
        # processes of the scoring process to be terminated successfully.
        os.setsid()
        if sigterm_on_parent_death:
            sigterm_on_parent_death()

    if os.name != "nt":
        proc = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            universal_newlines=True,
            env=env,
            preexec_fn=preexec_fn,
        )
    else:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            universal_newlines=True,
            env=env,
            # On Windows, `os.setsid` and `preexec_fn` are unavailable
            # `mlflow models serve` command creates several sub-processes,
            # In order to kill them easily via the root process,
            # set flag `CREATE_NEW_PROCESS_GROUP`, see
            # https://stackoverflow.com/questions/47016723/windows-equivalent-for-spawning-and-killing-separate-process-group-in-python-3
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    return proc


def kill_server(proc):
    if proc.poll() is None:
        # Terminate the process group containing the scoring process.
        # This will terminate all child processes of the scoring process
        if os.name != "nt":
            pgrp = os.getpgid(proc.pid)
            os.killpg(pgrp, signal.SIGTERM)
        else:
            # https://stackoverflow.com/questions/47016723/windows-equivalent-for-spawning-and-killing-separate-process-group-in-python-3
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.kill()
