import requests
import time
from mlflow.pyfunc import scoring_server
import json


class ScoringServerClient:
    def __init__(self, host, port):
        self.url_prefix = f"http://{host}:{port}"

    def ping(self):
        ping_status = requests.get(url=self.url_prefix + "/ping")
        if ping_status.status_code != 200:
            raise Exception(f"ping failed (error code {ping_status.status_code})")

    def wait_server_ready(self, timeout=30, scoring_server_proc=None):
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
            if scoring_server_proc is not None:
                return_code = scoring_server_proc.poll()
                if return_code is not None:
                    raise RuntimeError(f"Server process already exit with returncode {return_code}")
        raise RuntimeError("Wait scoring server ready timeout.")

    def invoke(self, data):
        """
        Invoke inference on input data. The input data must be pandas dataframe or json instance.
        """
        content_type = scoring_server.CONTENT_TYPE_JSON
        post_data = json.dumps(scoring_server._get_jsonable_obj(data, pandas_orient="split"))

        response = requests.post(
            url=self.url_prefix + "/invocations",
            data=post_data,
            headers={"Content-Type": content_type},
        )

        if response.status_code != 200:
            raise Exception(
                f"Invocation failed (error code {response.status_code}, response: {response.text})"
            )

        return scoring_server.infer_and_parse_json_input(response.text)
