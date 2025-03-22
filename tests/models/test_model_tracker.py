from sklearn.linear_model import LogisticRegression

import mlflow
from mlflow.models.model_tracker import _MODEL_TRACKER


def test_model_tracker():
    model = LogisticRegression().fit([[0, 1], [1, 0]], [0, 1])

    with mlflow.start_run():
        info_1 = mlflow.sklearn.log_model(model, "model")

    model_1 = mlflow.sklearn.load_model(info_1.model_uri)
    assert _MODEL_TRACKER.get_id(model_1) == info_1.model_id

    with mlflow.start_run():
        info_2 = mlflow.sklearn.log_model(model, "model")

    model_2 = mlflow.sklearn.load_model(info_2.model_uri)
    assert _MODEL_TRACKER.get_id(model_2) == info_2.model_id

    assert _MODEL_TRACKER.get_id(model_1) == info_1.model_id

    _MODEL_TRACKER.reset()
    assert _MODEL_TRACKER.get_id(model_1) is None
    assert _MODEL_TRACKER.get_id(model_2) is None
