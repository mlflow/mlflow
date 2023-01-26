import requests
import time
import json
import tempfile
import logging
import uuid
from pathlib import Path
from abc import ABC, abstractmethod


from mlflow.pyfunc import scoring_server

from mlflow.exceptions import MlflowException
from mlflow.utils.proto_json_utils import dump_input_data
from mlflow.deployments import PredictionsResponse

_logger = logging.getLogger(__name__)


class BaseScoringServerClient(ABC):
    @abstractmethod
    def wait_server_ready(self, timeout=30, scoring_server_proc=None):
        """
        Wait until the scoring server is ready to accept requests.
        """

    @abstractmethod
    def invoke(self, data):
        """
        Invoke inference on input data. The input data must be pandas dataframe or numpy array or
        a dict of numpy arrays.
        """


class ScoringServerClient(BaseScoringServerClient):
    def __init__(self, host, port):
        self.url_prefix = f"http://{host}:{port}"

    def ping(self):
        ping_status = requests.get(url=self.url_prefix + "/ping")
        if ping_status.status_code != 200:
            raise Exception(f"ping failed (error code {ping_status.status_code})")

    def get_version(self):
        resp_status = requests.get(url=self.url_prefix + "/version")
        if resp_status.status_code != 200:
            raise Exception(f"version failed (error code {resp_status.status_code})")
        return resp_status.text

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
        response = requests.post(
            url=self.url_prefix + "/invocations",
            data=dump_input_data(data),
            headers={"Content-Type": scoring_server.CONTENT_TYPE_JSON},
        )
        if response.status_code != 200:
            raise Exception(
                f"Invocation failed (error code {response.status_code}, response: {response.text})"
            )
        return PredictionsResponse.from_json(response.text)


class StdinScoringServerClient(BaseScoringServerClient):
    def __init__(self, process):
        self.process = process
        self.tmpdir = Path(tempfile.mkdtemp())
        self.output_json = self.tmpdir.joinpath("output.json")

    def wait_server_ready(self, timeout=30, scoring_server_proc=None):
        return_code = self.process.poll()
        if return_code is not None:
            raise RuntimeError(f"Server process already exit with returncode {return_code}")

    def invoke(self, data):
        """
        Invoke inference on input data. The input data must be pandas dataframe or numpy array or
        a dict of numpy arrays.
        """
        if not self.output_json.exists():
            self.output_json.touch()

        request_id = str(uuid.uuid4())
        request = {
            "id": request_id,
            "data": dump_input_data(data),
            "output_file": str(self.output_json),
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()

        begin_time = time.time()
        while True:
            _logger.info("Waiting for scoring to complete...")
            try:
                with self.output_json.open(mode="r+") as f:
                    resp = PredictionsResponse.from_json(f.read())
                    if resp.get("id") == request_id:
                        f.truncate(0)
                        return resp
            except Exception as e:
                _logger.debug("Exception while waiting for scoring to complete: %s", e)
            if time.time() - begin_time > 60:
                raise MlflowException("Scoring timeout")
            time.sleep(1)
