import argparse
import inspect
import json
import logging
import sys

import mlflow
from mlflow.pyfunc import scoring_server
from mlflow.pyfunc.model import _log_warning_if_params_not_in_predict_signature

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--model-uri")
args = parser.parse_args()

_logger.info("Loading model from %s", args.model_uri)
model = mlflow.pyfunc.load_model(args.model_uri)
input_schema = model.metadata.get_input_schema()
_logger.info("Loaded model")

_logger.info("Waiting for request")
for line in sys.stdin:
    _logger.info("Received request")
    request = json.loads(line)

    _logger.info("Parsing input data")
    data = request["data"]
    data, params = scoring_server._split_data_and_params(data)
    data = scoring_server.infer_and_parse_data(data, input_schema)

    _logger.info("Making predictions")
    if inspect.signature(model.predict).parameters.get("params"):
        preds = model.predict(data, params=params)
    else:
        _log_warning_if_params_not_in_predict_signature(_logger, params)
        preds = model.predict(data)

    _logger.info("Writing predictions")
    with open(request["output_file"], "a") as f:
        scoring_server.predictions_to_json(preds, f, {"id": request["id"]})

    _logger.info("Done")
