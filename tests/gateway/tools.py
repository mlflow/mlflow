import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple
from unittest import mock

import aiohttp
import requests
import transformers
import uvicorn
import yaml
from sentence_transformers import SentenceTransformer

import mlflow
from mlflow.gateway import app
from mlflow.gateway.utils import kill_child_processes

from tests.helper_functions import _get_mlflow_home, _start_scoring_proc, get_safe_port


class Gateway:
    def __init__(self, config_path: str | Path, *args, **kwargs):
        self.port = get_safe_port()
        self.host = "localhost"
        self.url = f"http://{self.host}:{self.port}"
        self.workers = 2
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "gateway",
                "start",
                "--config-path",
                config_path,
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--workers",
                str(self.workers),
            ],
            *args,
            **kwargs,
        )
        self.wait_until_ready()

    def wait_until_ready(self) -> None:
        s = time.time()
        while time.time() - s < 10:
            try:
                if self.get("health").ok:
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)

        raise Exception("Gateway failed to start")

    def wait_reload(self) -> None:
        """
        Should be called after we update a gateway config file in tests to ensure
        that the gateway service has reloaded the config.
        """
        time.sleep(self.workers)

    def request(self, method: str, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return requests.request(method, f"{self.url}/{path}", *args, **kwargs)

    def get(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.request("GET", path, *args, **kwargs)

    def assert_health(self):
        assert self.get("health").ok

    def post(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.request("POST", path, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        kill_child_processes(self.process.pid)
        self.process.terminate()
        self.process.wait()


def save_yaml(path, conf):
    path.write_text(yaml.safe_dump(conf))


class MockAsyncResponse:
    def __init__(self, data: dict[str, Any], status: int = 200):
        # Extract status and headers from data, if present
        self.status = status
        self.headers = data.pop("headers", {"Content-Type": "application/json"})

        # Save the rest of the data as content
        self._content = data

    def raise_for_status(self) -> None:
        if 400 <= self.status < 600:
            raise aiohttp.ClientResponseError(None, None, status=self.status)

    async def json(self) -> dict[str, Any]:
        return self._content

    async def text(self) -> str:
        return json.dumps(self._content)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        pass


class MockAsyncStreamingResponse:
    def __init__(self, data: list[bytes], headers: dict[str, str] | None = None, status: int = 200):
        self.status = status
        self.headers = headers
        self._content = data

    def raise_for_status(self) -> None:
        if 400 <= self.status < 600:
            raise aiohttp.ClientResponseError(None, None, status=self.status)

    async def _async_content(self):
        for line in self._content:
            yield line

    @property
    def content(self):
        return self._async_content()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        pass


class MockHttpClient(mock.Mock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return


def mock_http_client(mock_response: MockAsyncResponse | MockAsyncStreamingResponse):
    mock_http_client = MockHttpClient()
    mock_http_client.post = mock.Mock(return_value=mock_response)
    return mock_http_client


class UvicornGateway:
    # This test utility class is used to validate the internal functionality of the
    # AI Gateway within-process so that the provider endpoints can be mocked,
    # allowing a nearly end-to-end validation of the entire AI Gateway stack.
    # NB: this implementation should only be used for integration testing. Unit tests that
    # require validation of the AI Gateway server should use the `Gateway` implementation in
    # this module which executes the uvicorn server through gunicorn as a process manager.
    def __init__(self, config_path: str | Path, *args, **kwargs):
        self.port = get_safe_port()
        self.host = "127.0.0.1"
        self.url = f"http://{self.host}:{self.port}"
        self.config_path = config_path
        self.server = None
        self.loop = None
        self.thread = None
        self.stop_event = threading.Event()

    def start_server(self):
        uvicorn_app = app.create_app_from_path(self.config_path)

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        config = uvicorn.Config(
            app=uvicorn_app,
            host=self.host,
            port=self.port,
            lifespan="on",
            loop="auto",
            log_level="info",
        )
        self.server = uvicorn.Server(config)

        def run():
            self.loop.run_until_complete(self.server.serve())

        self.thread = threading.Thread(target=run)
        self.thread.start()

    def request(self, method: str, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return requests.request(method, f"{self.url}/{path}", *args, **kwargs)

    def get(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.request("GET", path, *args, **kwargs)

    def assert_health(self):
        assert self.get("health").ok

    def post(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.request("POST", path, *args, **kwargs)

    def stop(self):
        if self.server is not None:
            self.server.should_exit = True  # Instruct the uvicorn server to stop
            self.stop_event.wait()  # Wait for the server to actually stop
            self.thread.join()  # block until thread termination
            self.server = None
            self.loop = None
            self.thread = None

    def __enter__(self):
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop the server and the thread
        if self.server is not None:
            self.server.should_exit = True
        self.thread.join()


class ServerInfo(NamedTuple):
    pid: int
    url: str


def log_sentence_transformers_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    artifact_path = "gen_model"

    with mlflow.start_run():
        model_info = mlflow.sentence_transformers.log_model(
            model,
            name=artifact_path,
        )
        return model_info.model_uri


def log_completions_transformers_model():
    architecture = "distilbert-base-uncased"

    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    model = transformers.AutoModelForMaskedLM.from_pretrained(architecture)
    pipe = transformers.pipeline(task="fill-mask", model=model, tokenizer=tokenizer)

    inference_params = {"top_k": 1}

    signature = mlflow.models.infer_signature(
        ["test1 [MASK]", "[MASK] test2"],
        mlflow.transformers.generate_signature_output(pipe, ["test3 [MASK]"]),
        inference_params,
    )

    artifact_path = "mask_model"

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            pipe,
            name=artifact_path,
            signature=signature,
        )
        return model_info.model_uri


def start_mlflow_server(port, model_uri):
    server_url = f"http://127.0.0.1:{port}"

    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    env.update(MLFLOW_TRACKING_URI=mlflow.get_tracking_uri())
    env.update(MLFLOW_HOME=_get_mlflow_home())
    scoring_cmd = [
        "mlflow",
        "models",
        "serve",
        "-m",
        model_uri,
        "-p",
        str(port),
        "--install-mlflow",
        "--no-conda",
    ]

    server_pid = _start_scoring_proc(cmd=scoring_cmd, env=env, stdout=sys.stdout, stderr=sys.stdout)

    ping_status = None
    for i in range(120):
        time.sleep(1)
        try:
            ping_status = requests.get(url=f"{server_url}/ping")
            if ping_status.status_code == 200:
                break
        except Exception:
            pass
    if ping_status is None or ping_status.status_code != 200:
        raise Exception("Could not start mlflow serving instance.")

    return ServerInfo(pid=server_pid, url=server_url)


def stop_mlflow_server(server_pid):
    process_group = os.getpgid(server_pid.pid)
    os.killpg(process_group, signal.SIGTERM)
