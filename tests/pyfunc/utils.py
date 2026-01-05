import json
import os
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

import mlflow
from mlflow.pyfunc import scoring_server

if TYPE_CHECKING:
    import httpx


def score_model_in_process(model_uri: str, data: str, content_type: str) -> "httpx.Response":
    """Score a model using in-process FastAPI TestClient (faster than subprocess)."""
    import pandas as pd

    env_snapshot = os.environ.copy()
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        app = scoring_server.init(model)
        client = TestClient(app)

        # Convert DataFrame to JSON format if needed (matching RestEndpoint.invoke behavior)
        if isinstance(data, pd.DataFrame):
            if content_type == scoring_server.CONTENT_TYPE_CSV:
                data = data.to_csv(index=False)
            else:
                assert content_type == scoring_server.CONTENT_TYPE_JSON
                data = json.dumps({"dataframe_split": data.to_dict(orient="split")})
        elif not isinstance(data, (str, dict)):
            data = json.dumps({"instances": data})

        return client.post("/invocations", content=data, headers={"Content-Type": content_type})
    finally:
        os.environ.clear()
        os.environ.update(env_snapshot)
