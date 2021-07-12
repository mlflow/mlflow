import os
from mlflux.pyfunc import scoring_server
from mlflux.pyfunc import load_model


app = scoring_server.init(load_model(os.environ[scoring_server._SERVER_MODEL_PATH]))
