import os

from mlflow.pyfunc import scoring_server

app = scoring_server.init(
    scoring_server.load_model_with_mlflow_config(os.environ[scoring_server._SERVER_MODEL_PATH])
)
