import requests
import time
import json
import numpy as np
import pandas as pd

from mlflow.pyfunc import scoring_server

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.proto_json_utils import _DateTimeEncoder


class MlflowModelServerOutput(dict):
    """
    Represents the predictions and response metadata from a scoring API request sent to an
    MLflow Model Server, including Model Servers launched using MLflow Deployments.
    """

    def get_predictions(self, predictions_format="dataframe", dtype=None):
        """
        Get the predictions returned from the MLflow Model Server in the specified format.

        :param predictions_format: The format in which to return the predictions. Either
                                   ``"dataframe"`` or ``"ndarray"``.
        :param dtype: The NumPy datatype to which to coerce the predictions. Only used when
                      the ``"ndarray"`` ``predictions_format`` is specified.
        :throws: Exception if the predictions cannot be represented in the specified format.
        :return: The predictions, represented in the specified format.
        """
        if predictions_format == "dataframe":
            return pd.DataFrame(data=self["predictions"])
        elif predictions_format == "ndarray":
            return np.array(self["predictions"], dtype)
        else:
            raise MlflowException(
                f"Unrecognized predictions format: '{predictions_format}'",
                INVALID_PARAMETER_VALUE,
            )

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
