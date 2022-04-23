import warnings
import sys
import json

from bigml.api import BigML
from urllib.parse import urlparse
import mlflow
import mlflow.bigml

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def train_test_split(api, dataset):
    train_conf = {"seed": "bigml", "sample_rate": 0.8}
    test_conf = {"seed": "bigml", "sample_rate": 0.8, "out_of_bag": True}
    train_dataset = api.create_dataset(dataset, train_conf)
    api.ok(train_dataset)
    test_dataset = api.create_dataset(dataset, test_conf)
    api.ok(test_dataset)
    return train_dataset, test_dataset


def evaluation_metric(evaluation, metric_name):
    """Returns the corresponding evaluation metric"""
    return evaluation["object"]["result"]["model"][metric_name]


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Read the diabetes csv file from the URL
    csv_url = "http://static.bigml.com/csv/diabetes.csv"
    # create connection
    api = BigML()
    print("Connecting to BigML.")
    # create source
    source = api.create_source(csv_url)
    print("Creating Source from %s." % csv_url)
    api.ok(source)

    # create dataset
    dataset = api.create_dataset(source)
    print("Creating Dataset.")
    api.ok(dataset)

    # Split the data into training and test sets. (0.8, 0.2) split.
    print("Creating Train/Test Datasets.")
    train, test = train_test_split(api, dataset)
    model_conf = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}

    with mlflow.start_run():
        model = api.create_model(train, args=model_conf)
        print("Creating Model.")
        api.ok(model, query_string="limit=-1")
        evaluation = api.create_evaluation(model, test)
        print("Creating Evaluation.")
        api.ok(evaluation)
        print(
            "BigML model: %s\nconf: %s (%s)"
            % (model["object"]["name"], model["object"]["name_options"], model["resource"])
        )
        print("accuracy: %s" % evaluation_metric(evaluation, "accuracy"))
        print("precision: %s" % evaluation_metric(evaluation, "average_precision"))
        print("recall: %s" % evaluation_metric(evaluation, "average_recall"))

        mlflow.log_param("args", json.dumps(model_conf))
        mlflow.log_metric("accuracy", evaluation_metric(evaluation, "accuracy"))
        mlflow.log_metric("precision", evaluation_metric(evaluation, "average_precision"))
        mlflow.log_metric("recall", evaluation_metric(evaluation, "average_recall"))

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.bigml.log_model(model, "model", registered_model_name="BigML_Diabetes_dt")
        else:
            mlflow.bigml.log_model(model, "model")
