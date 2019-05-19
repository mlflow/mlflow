"""
This file declares a Flask app serving a pyfunc model that can be used by e.g. gunicorn.
The model is loaded from a path read from environment variable:
`mlflow.pyfunc.scoring_server.MLFLOW_MODEL_PATH`
"""
import os
from mlflow.pyfunc import scoring_server
from mlflow import pyfunc

app = scoring_server.init(pyfunc.load_pyfunc(os.environ[scoring_server.MLFLOW_MODEL_PATH]))
