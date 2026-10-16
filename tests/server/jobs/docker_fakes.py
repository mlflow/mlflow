"""A fake docker-py client for job container tests: no Docker needed."""

import json
from pathlib import Path

from mlflow.server.jobs import job_container as jc
from mlflow.server.jobs.utils import _SubprocessJobResult


class FakeContainer:
    def __init__(self, client, kwargs):
        self.id = f"container-{len(client.started)}"
        self.short_id = self.id
        self.kwargs = kwargs
        self.killed = False
        self.removed = False
        self.on_wait = client.on_wait

    def _output_dir(self):
        for host, spec in self.kwargs["volumes"].items():
            if spec["bind"] == jc._CONTAINER_OUTPUT_DIR:
                return Path(host)

    def wait(self, timeout=None):
        return self.on_wait(self)

    def logs(self, tail=None):
        self.logs_tail = tail
        return b"container log line"

    def kill(self):
        self.killed = True

    def remove(self, force=False):
        self.removed = True


class FakeContainers:
    def __init__(self, client):
        self._client = client

    def run(self, image, **kwargs):
        if self._client.on_run is not None:
            self._client.on_run()
        container = FakeContainer(self._client, kwargs)
        container.image = image
        if self._client.start_error is not None:
            # Like docker-py: the container is created, then starting it fails.
            self._client.leftovers.append(container)
            raise self._client.start_error
        self._client.started.append(container)
        return container

    def get(self, container_id):
        if self._client.get_error is not None:
            raise self._client.get_error
        return next(c for c in self._client.started if c.id == container_id)

    def list(self, all=False, filters=None):
        containers = list(self._client.leftovers)
        if filters and "label" in filters:
            key, _, value = filters["label"].partition("=")
            containers = [c for c in containers if c.kwargs.get("labels", {}).get(key) == value]
        return containers


class FakeClient:
    def __init__(self, on_wait=None):
        self.started = []
        self.leftovers = []
        self.on_wait = on_wait or succeed_with({"score": 1})
        self.start_error = None
        self.get_error = None
        self.on_run = None
        self.containers = FakeContainers(self)

    def ping(self):
        return True


def succeed_with(value):
    def on_wait(container):
        _SubprocessJobResult(succeeded=True, result=json.dumps(value)).dump(
            str(container._output_dir() / jc._RESULT_FILE)
        )
        return {"StatusCode": 0}

    return on_wait
