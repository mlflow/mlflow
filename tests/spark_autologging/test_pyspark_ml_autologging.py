import importlib
from collections import namedtuple

import mlflow
from mlflow.utils import (
    truncate_dict,
)
from mlflow.utils.validation import (
    MAX_PARAM_VAL_LENGTH,
    MAX_ENTITY_KEY_LENGTH,
)

from tests.autologging.fixtures import test_mode_off
from tests.spark_autologging.utils import spark_session  # pylint: disable=unused-import
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from mlflow.pyspark.ml import _get_estimator_param_map, _log_model_allowlist

MODEL_DIR = "model"


def truncate_param_dict(d):
    return truncate_dict(d, MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)


def stringify_dict_values(d):
    return {k: str(v) for k, v in d.items()}


def get_expected_class_tags(estimator):
    return {
        'estimator_name': estimator.__class__.__name__,
        'estimator_class': estimator.__class__.__module__ + "." + estimator.__class__.__name__,
    }


def get_run_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    # Ignore tags mlflow logs by default (e.g. "mlflow.user")
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id)]

    RunData = namedtuple('RunData', ['params', 'metrics', 'tags', 'artifacts'])
    return RunData(data.params, data.metrics, tags, artifacts)


def load_model_by_run_id(run_id):
    return mlflow.spark.load_model("runs:/{}/{}".format(run_id, MODEL_DIR))


def get_training_dataset(spark_session):
    return spark_session.createDataFrame(
        [(1.0, Vectors.dense(1.0)),
         (0.0, Vectors.sparse(1, [], []))] * 100,
        ["label", "features"])


def test_basic_estimator(spark_session):
    mlflow.pyspark.ml.autolog()
    lr = LinearRegression()
    training_dataset = get_training_dataset(spark_session)
    with mlflow.start_run() as run:
        lr_model = lr.fit(training_dataset)
    run_id = run.info.run_id
    run_data = get_run_data(run_id)
    assert run_data.params == \
           truncate_param_dict(stringify_dict_values(_get_estimator_param_map(lr)))
    assert run_data.tags == get_expected_class_tags(lr)
    assert MODEL_DIR in run_data.artifacts
    loaded_model = load_model_by_run_id(run_id)
    assert loaded_model.stages[0].uid == lr_model.uid


def test_allowlist_file():
    for estimator_class in _log_model_allowlist:
        pieces = estimator_class.split('.')
        module_name = '.'.join(pieces[:-1])
        class_name = pieces[-1]
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
        except:
            assert False, 'The {estimator_class} in log_model_allowlist does not exists' \
                .format(estimator_class=estimator_class)
