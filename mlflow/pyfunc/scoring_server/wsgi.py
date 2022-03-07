import os
from mlflow.pyfunc import scoring_server
from mlflow.pyfunc import load_model, _load_model_from_local_path


if scoring_server._SERVER_MODEL_LOCAL_PATH in os.environ:
    model = _load_model_from_local_path(os.environ[scoring_server._SERVER_MODEL_LOCAL_PATH])
else:
    model = load_model(os.environ[scoring_server._SERVER_MODEL_PATH])

app = scoring_server.init(model)
