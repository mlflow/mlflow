import requests
import time
from mlflow.pyfunc import scoring_server
import subprocess
import os
import sys
import signal


class ScoringServerClient:

    def __init__(self, host, port):
        self.url_prefix = f"http://{host}:{port}"

    def ping(self):
        ping_status = requests.get(url=self.url_prefix + "/ping")
        print("connection attempt", i, "server is up! ping status", ping_status)
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
        raise RuntimeError('Wait scoring server ready timeout.')

    def invoke(self, data, pandas_orient="records"):
        import pandas as pd

        content_type_list = []
        if type(data) == pd.DataFrame:
            content_type_list.append(scoring_server.CONTENT_TYPE_JSON)

            if pandas_orient == "records":
                content_type_list.append(scoring_server.CONTENT_TYPE_FORMAT_RECORDS_ORIENTED)
            elif pandas_orient == "split":
                content_type_list.append(scoring_server.CONTENT_TYPE_FORMAT_SPLIT_ORIENTED)
            else:
                raise Exception(
                    "Unexpected pandas_orient for Pandas dataframe input %s" % pandas_orient
                )
            data = data.to_json(orient=pandas_orient)
        else:
            raise RuntimeError("Unsupported data type.")

        response = requests.post(
            url=self.url_prefix + "/invocations",
            data=data,
            headers={"Content-Type": "; ".join(content_type_list)},
        )

        if response.status_code != 200:
            raise Exception(
                f"Invocation failed (error code {response.status_code}, response: {response.text})"
            )

        return scoring_server._get_jsonable_obj(response.text, pandas_orient)


def start_server(
        server_port, local_model_path,
        host='127.0.0.1', num_workers=1,
        env=None, stdout=sys.stdout, stderr=sys.stderr
):
    cmd = [
        "mlflow", "models", "serve", "-m", local_model_path,
        "-h", host,
        "-p", str(server_port), "-w", str(num_workers), "--install-mlflow",
    ]
    if os.name != "nt":
        return subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            universal_newlines=True,
            env=env,
            # Assign the scoring process to a process group. All child processes of the
            # scoring process will be assigned to this group as well. This allows child
            # processes of the scoring process to be terminated successfully
            preexec_fn=os.setsid,
        )
    else:
        return subprocess.Popen(
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
