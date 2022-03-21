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

    def invoke(self, data, pandas_orient="records"):
        """
        Invoke inference on input data. The input data must be pandas dataframe or json instance.
        """

        import pandas as pd

        content_type_list = [scoring_server.CONTENT_TYPE_JSON]
        if isinstance(data, pd.DataFrame):
            if pandas_orient == "records":
                content_type_list.append(scoring_server.CONTENT_TYPE_FORMAT_RECORDS_ORIENTED)
            elif pandas_orient == "split":
                content_type_list.append(scoring_server.CONTENT_TYPE_FORMAT_SPLIT_ORIENTED)
            else:
                raise Exception(
                    "Unexpected pandas_orient for Pandas dataframe input %s" % pandas_orient
                )
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

        return scoring_server.infer_and_parse_json_input(response.text)


def prepare_env(local_model_path, stdout=sys.stdout, stderr=sys.stderr):
    cmd = [
        "mlflow",
        "models",
        "prepare-env",
        "-m",
        local_model_path,
    ]
    return subprocess.run(cmd, stdout=stdout, stderr=stderr, universal_newlines=True, check=True)
