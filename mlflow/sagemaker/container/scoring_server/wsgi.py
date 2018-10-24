from mlflow.pyfunc import scoring_server
from mlflow import pyfunc


def app(model_path):
    return scoring_server.init(pyfunc.load_pyfunc(model_path))
