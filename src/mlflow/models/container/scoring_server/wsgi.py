from mlflow import pyfunc
from mlflow.pyfunc import scoring_server

app = scoring_server.init(pyfunc.load_model("/opt/ml/model/"))
