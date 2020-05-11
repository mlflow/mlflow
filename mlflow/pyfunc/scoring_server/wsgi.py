import os

from mlflow.models.flavor_backend import SchemaEnforcement
from mlflow.pyfunc import scoring_server
from mlflow.pyfunc import load_model


def build_app(schema_enforcement: SchemaEnforcement.NONE):
    return scoring_server.init(load_model(os.environ[scoring_server._SERVER_MODEL_PATH]),
                               schema_enforcement)


app = build_app()
