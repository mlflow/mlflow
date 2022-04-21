import warnings
import json

import mlflow.bigml

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Read the model JSON file from the file
    model_file = "bigml_logistic/logistic_regression.json"
    with open(model_file) as handler:
        model = json.load(handler)

    with mlflow.start_run():
        print("Registering BigML logistic regression: %s\nconf: %s (%s)" % (
            model["object"]["name"],
            model["object"]["name_options"], model["resource"]))
        mlflow.bigml.log_model(model, "model")
        """
        Testing example:
        curl -d '{"columns":["plasma glucose"], "data":[[90], [220]]}' \
             -H 'Content-Type: application/json; format=pandas-split' \
             -X POST localhost:5000/invocations
        """
