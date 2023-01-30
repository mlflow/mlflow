import argparse
import sys
import json
import logging

import mlflow
from mlflow.pyfunc import scoring_server

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
    data = scoring_server.infer_and_parse_json_input(data, input_schema)

    _logger.info("Making predictions")
    preds = model.predict(data)

    _logger.info("Writing predictions")
    with open(request["output_file"], "a") as f:
        scoring_server.predictions_to_json(preds, f, {"id": request["id"]})

    _logger.info("Done")
