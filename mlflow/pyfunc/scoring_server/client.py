import requests
import time
from mlflow.pyfunc import scoring_server


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
        import numpy as np
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
                    "Unexpected content type for Pandas dataframe input %s" % content_type
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

