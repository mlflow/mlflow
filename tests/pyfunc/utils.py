import os
from typing import TYPE_CHECKING

import mlflow
from mlflow.pyfunc import scoring_server

if TYPE_CHECKING:
    import httpx


def score_model_in_process(model_uri: str, data: str, content_type: str) -> "httpx.Response":
    """Score a model using in-process FastAPI TestClient (faster than subprocess)."""
    from fastapi.testclient import TestClient

    env_snapshot = os.environ.copy()
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        app = scoring_server.init(model)
        client = TestClient(app)
        return client.post("/invocations", content=data, headers={"Content-Type": content_type})
    finally:
        os.environ.clear()
        os.environ.update(env_snapshot)
