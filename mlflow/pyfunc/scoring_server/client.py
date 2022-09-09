import requests
import time
import json
import numpy as np
import pandas as pd

from mlflow.pyfunc import scoring_server

from mlflow.exceptions import MlflowException
from mlflow.utils.proto_json_utils import _DateTimeEncoder


class MlflowModelServerOutput(dict):
    """Predictions returned from the MLflow scoring server. The predictions are presented as a
    dictionary with convenience methods to access the predictions parsed as higher level type."""

    @classmethod
    def from_raw_json(cls, json_str):
        try:
            parsed_response = json.loads(json_str)
        except Exception as ex:
            raise MlflowException(f"Model response is not a valid json. Error {ex}")
        if not isinstance(parsed_response, dict) or "predictions" not in parsed_response:
            raise MlflowException(
                "Invalid predictions data. Prediction object must be a dictionary "
                "with 'predictions' field."
            )
        return MlflowModelServerOutput(parsed_response)

    def get_predictions_dataframe(self):
        """Get the predictions returned from the server as pandas.DataFrame. this method will fail
        if the returned predictions is not a valid DataFrame representation."""
        return pd.DataFrame(data=self["predictions"])

    def get_predictions_nparray(self, dtype=None):
        """Get the predictions returned from the server as a numpy array. this method will fail
        if the returned predictions is not a valid numpy array"""
        return np.array(self["predictions"], dtype)


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
        Invoke inference on input data. The input data must be pandas dataframe or numpy array or
        a dict of numpy arrays.
        """
        content_type = scoring_server.CONTENT_TYPE_JSON

        def get_jsonable_input(name, data):
            if isinstance(data, np.ndarray):
                return data.tolist()
            else:
                raise MlflowException(f"Incompatible input type:{type(data)} for input {name}.")

        if isinstance(data, pd.DataFrame):
            post_data = {"dataframe_split": data.to_dict(orient="split")}
        elif isinstance(data, dict):
            post_data = {"inputs": {k: get_jsonable_input(k, v) for k, v in data}}
        elif isinstance(data, np.ndarray):
            post_data = ({"inputs": data.tolist()},)
        else:
            post_data = data
        if not isinstance(post_data, str):
            post_data = json.dumps(post_data, cls=_DateTimeEncoder)
        response = requests.post(
            url=self.url_prefix + "/invocations",
            data=post_data,
            headers={"Content-Type": content_type},
        )
        if response.status_code != 200:
            raise Exception(
                f"Invocation failed (error code {response.status_code}, response: {response.text})"
            )
        return MlflowModelServerOutput.from_raw_json(response.text)
