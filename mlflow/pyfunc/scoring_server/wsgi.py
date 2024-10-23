import os
import json

from mlflow.pyfunc import load_model, scoring_server

extra_kwargs = {}
if model_config_json := os.environ[scoring_server.SERVING_MODEL_CONFIG]:
    extra_kwargs["model_config"] = json.loads(model_config_json)

app = scoring_server.init(
    load_model(
        os.environ[scoring_server._SERVER_MODEL_PATH],
        **extra_kwargs,
    )
)
