import warnings
import json
import os

import mlflow.bigml

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Read the ensemble JSON file from the file and all the models
    # it uses
    models = []
    model_file = "bigml_ensemble/ensemble.json"
    path = os.path.dirname(model_file)
    with open(model_file) as handler:
        model = json.load(handler)
        models.append(model)
    for model_id in model["object"]["models"]:
        filename = os.path.join(path, model_id.replace("/", "_"))
        with open(filename) as component_handler:
            models.append(json.load(component_handler))

    with mlflow.start_run():
        print("Registering BigML ensemble: %s\nconf: %s (%s)" % (
            model["object"]["name"],
            model["object"]["name_options"], model["resource"]))
        mlflow.bigml.log_model(models, "model")
