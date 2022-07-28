from collections import namedtuple
import pytest
import numpy as np
import pandas as pd
import os
from unittest import mock
import yaml

import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from mlflow import pyfunc
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.paddle
from mlflow.models import Model
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

from tests.helper_functions import (
    pyfunc_serve_and_score_model,
    _assert_pip_requirements,
    _compare_logged_code_paths,
    PROTOBUF_REQUIREMENT,
)


ModelWithData = namedtuple("ModelWithData", ["model", "inference_dataframe"])


def get_dataset():
    X, y = load_diabetes(return_X_y=True)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_min_max = min_max_scaler.fit_transform(X)
    X_normalized = preprocessing.scale(X_min_max, with_std=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return np.concatenate((X_train, y_train), axis=1), np.concatenate((X_test, y_test), axis=1)


@pytest.fixture
def pd_model():
    class Regressor(paddle.nn.Layer):
        def __init__(self, in_features):
            super().__init__()
            self.fc_ = Linear(in_features=in_features, out_features=1)

        @paddle.jit.to_static
        def forward(self, inputs):  # pylint: disable=arguments-differ
            return self.fc_(inputs)

    training_data, test_data = get_dataset()
    model = Regressor(training_data.shape[1] - 1)
    model.train()
    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

    EPOCH_NUM = 10
    BATCH_SIZE = 10

    for _ in range(EPOCH_NUM):
        np.random.shuffle(training_data)
        mini_batches = [
            training_data[k : k + BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)
        ]
        for mini_batch in mini_batches:
            x = np.array(mini_batch[:, :-1]).astype("float32")
            y = np.array(mini_batch[:, -1:]).astype("float32")
            house_features = paddle.to_tensor(x)
            prices = paddle.to_tensor(y)
            predicts = model(house_features)
            loss = F.square_error_cost(predicts, label=prices)
            avg_loss = paddle.mean(loss)

            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    np_test_data = np.array(test_data).astype("float32")
    return ModelWithData(model=model, inference_dataframe=np_test_data[:, :-1])


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture
def pd_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["paddle", "pytest"])
    return conda_env


def test_model_save_load(pd_model, model_path):
    mlflow.paddle.save_model(pd_model=pd_model.model, path=model_path)

    reloaded_pd_model = mlflow.paddle.load_model(model_uri=model_path)
    reloaded_pyfunc = pyfunc.load_model(model_uri=model_path)

    np.testing.assert_array_almost_equal(
        pd_model.model(pd_model.inference_dataframe),
        reloaded_pyfunc.predict(pd_model.inference_dataframe),
        decimal=5,
    )

    np.testing.assert_array_almost_equal(
        reloaded_pd_model(pd_model.inference_dataframe),
        reloaded_pyfunc.predict(pd_model.inference_dataframe),
        decimal=5,
    )


def test_model_load_from_remote_uri_succeeds(pd_model, model_path, mock_s3_bucket):
    mlflow.paddle.save_model(pd_model=pd_model.model, path=model_path)

    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    reloaded_model = mlflow.paddle.load_model(model_uri=model_uri)
    np.testing.assert_array_almost_equal(
        pd_model.model(pd_model.inference_dataframe),
        reloaded_model(pd_model.inference_dataframe),
        decimal=5,
    )


def test_model_log(pd_model, model_path, tmpdir):
    model = pd_model.model
    try:
        artifact_path = "model"
        conda_env = os.path.join(tmpdir, "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["paddle"])

        model_info = mlflow.paddle.log_model(
            pd_model=model, artifact_path=artifact_path, conda_env=conda_env
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )
        assert model_info.model_uri == model_uri

        reloaded_pd_model = mlflow.paddle.load_model(model_uri=model_uri)
        np.testing.assert_array_almost_equal(
            model(pd_model.inference_dataframe),
            reloaded_pd_model(pd_model.inference_dataframe),
            decimal=5,
        )

        model_path = _download_artifact_from_uri(artifact_uri=model_uri)
        model_config = Model.load(os.path.join(model_path, "MLmodel"))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
        assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
        env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]
        assert os.path.exists(os.path.join(model_path, env_path))
    finally:
        mlflow.end_run()


def test_log_model_calls_register_model(pd_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.paddle.log_model(
            pd_model=pd_model.model,
            artifact_path=artifact_path,
            registered_model_name="AdsModel1",
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )
        mlflow.register_model.assert_called_once_with(
            model_uri, "AdsModel1", await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )


def test_log_model_no_registered_model_name(pd_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.paddle.log_model(pd_model=pd_model.model, artifact_path=artifact_path)
        mlflow.register_model.assert_not_called()


def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    pd_model, model_path, pd_custom_env
):
    mlflow.paddle.save_model(pd_model=pd_model.model, path=model_path, conda_env=pd_custom_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pd_custom_env

    with open(pd_custom_env, "r") as f:
        pd_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pd_custom_env_parsed


def test_model_save_accepts_conda_env_as_dict(pd_model, model_path):
    conda_env = dict(mlflow.paddle.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.paddle.save_model(pd_model=pd_model.model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(pd_model, pd_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.paddle.log_model(
            pd_model=pd_model.model, artifact_path=artifact_path, conda_env=pd_custom_env
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pd_custom_env

    with open(pd_custom_env, "r") as f:
        pd_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pd_custom_env_parsed


def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    pd_model, model_path
):
    mlflow.paddle.save_model(pd_model=pd_model.model, path=model_path)
    _assert_pip_requirements(model_path, mlflow.paddle.get_default_pip_requirements())


def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    pd_model,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.paddle.log_model(pd_model=pd_model.model, artifact_path=artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)
    _assert_pip_requirements(model_uri, mlflow.paddle.get_default_pip_requirements())


@pytest.fixture(scope="module")
def get_dataset_built_in_high_level_api():
    train_dataset = paddle.text.datasets.UCIHousing(mode="train")
    eval_dataset = paddle.text.datasets.UCIHousing(mode="test")
    return train_dataset, eval_dataset


class UCIHousing(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc_ = paddle.nn.Linear(13, 1, None)

    def forward(self, inputs):  # pylint: disable=arguments-differ
        pred = self.fc_(inputs)
        return pred


@pytest.fixture
def pd_model_built_in_high_level_api(get_dataset_built_in_high_level_api):
    train_dataset, test_dataset = get_dataset_built_in_high_level_api

    model = paddle.Model(UCIHousing())
    optim = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    model.prepare(optim, paddle.nn.MSELoss())

    model.fit(train_dataset, epochs=6, batch_size=8, verbose=1)

    return ModelWithData(model=model, inference_dataframe=test_dataset)


def test_model_save_load_built_in_high_level_api(pd_model_built_in_high_level_api, model_path):
    model = pd_model_built_in_high_level_api.model
    test_dataset = pd_model_built_in_high_level_api.inference_dataframe
    mlflow.paddle.save_model(pd_model=model, path=model_path)

    reloaded_pd_model = mlflow.paddle.load_model(model_uri=model_path)
    reloaded_pyfunc = pyfunc.load_model(model_uri=model_path)

    low_level_test_dataset = [x[0] for x in test_dataset]

    np.testing.assert_array_almost_equal(
        np.array(model.predict(test_dataset)).squeeze(),
        np.array(reloaded_pyfunc.predict(np.array(low_level_test_dataset))).squeeze(),
        decimal=5,
    )

    np.testing.assert_array_almost_equal(
        np.array(reloaded_pd_model(np.array(low_level_test_dataset))).squeeze(),
        np.array(reloaded_pyfunc.predict(np.array(low_level_test_dataset))).squeeze(),
        decimal=5,
    )


def test_model_built_in_high_level_api_load_from_remote_uri_succeeds(
    pd_model_built_in_high_level_api, model_path, mock_s3_bucket
):
    model = pd_model_built_in_high_level_api.model
    test_dataset = pd_model_built_in_high_level_api.inference_dataframe
    mlflow.paddle.save_model(pd_model=model, path=model_path)

    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    reloaded_model = mlflow.paddle.load_model(model_uri=model_uri)

    low_level_test_dataset = [x[0] for x in test_dataset]

    np.testing.assert_array_almost_equal(
        np.array(model.predict(test_dataset)).squeeze(),
        np.array(reloaded_model(np.array(low_level_test_dataset))).squeeze(),
        decimal=5,
    )


def test_model_built_in_high_level_api_log(pd_model_built_in_high_level_api, model_path, tmpdir):
    model = pd_model_built_in_high_level_api.model
    test_dataset = pd_model_built_in_high_level_api.inference_dataframe
    try:
        artifact_path = "model"
        conda_env = os.path.join(tmpdir, "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["paddle"])

        mlflow.paddle.log_model(pd_model=model, artifact_path=artifact_path, conda_env=conda_env)
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )

        reloaded_pd_model = mlflow.paddle.load_model(model_uri=model_uri)
        low_level_test_dataset = [x[0] for x in test_dataset]
        np.testing.assert_array_almost_equal(
            np.array(model.predict(test_dataset)).squeeze(),
            np.array(reloaded_pd_model(np.array(low_level_test_dataset))).squeeze(),
            decimal=5,
        )

        model_path = _download_artifact_from_uri(artifact_uri=model_uri)
        model_config = Model.load(os.path.join(model_path, "MLmodel"))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
        assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
        env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]
        assert os.path.exists(os.path.join(model_path, env_path))
    finally:
        mlflow.end_run()


@pytest.fixture
def model_retrain_path(tmpdir):
    return os.path.join(str(tmpdir), "model_retrain")


@pytest.mark.allow_infer_pip_requirements_fallback
def test_model_retrain_built_in_high_level_api(
    pd_model_built_in_high_level_api,
    model_path,
    model_retrain_path,
    get_dataset_built_in_high_level_api,
):
    model = pd_model_built_in_high_level_api.model
    mlflow.paddle.save_model(pd_model=model, path=model_path, training=True)

    training_dataset, test_dataset = get_dataset_built_in_high_level_api

    model_retrain = paddle.Model(UCIHousing())
    model_retrain = mlflow.paddle.load_model(model_uri=model_path, model=model_retrain)
    optim = paddle.optimizer.Adam(learning_rate=0.015, parameters=model.parameters())
    model_retrain.prepare(optim, paddle.nn.MSELoss())

    model_retrain.fit(training_dataset, epochs=6, batch_size=8, verbose=1)

    mlflow.paddle.save_model(pd_model=model_retrain, path=model_retrain_path, training=False)

    with pytest.raises(TypeError, match="This model can't be loaded"):
        mlflow.paddle.load_model(model_uri=model_retrain_path, model=model_retrain)

    error_model = 0
    error_model_type = type(error_model)
    with pytest.raises(
        TypeError,
        match="Invalid object type `{}` for `model`, must be `paddle.Model`".format(
            error_model_type
        ),
    ):
        mlflow.paddle.load_model(model_uri=model_retrain_path, model=error_model)

    reloaded_pd_model = mlflow.paddle.load_model(model_uri=model_retrain_path)
    reloaded_pyfunc = pyfunc.load_model(model_uri=model_retrain_path)
    low_level_test_dataset = [x[0] for x in test_dataset]

    np.testing.assert_array_almost_equal(
        np.array(model_retrain.predict(test_dataset)).squeeze(),
        np.array(reloaded_pyfunc.predict(np.array(low_level_test_dataset))).squeeze(),
        decimal=5,
    )

    np.testing.assert_array_almost_equal(
        np.array(reloaded_pd_model(np.array(low_level_test_dataset))).squeeze(),
        np.array(reloaded_pyfunc.predict(np.array(low_level_test_dataset))).squeeze(),
        decimal=5,
    )


def test_log_model_built_in_high_level_api(
    pd_model_built_in_high_level_api, model_path, tmpdir, get_dataset_built_in_high_level_api
):
    model = pd_model_built_in_high_level_api.model
    test_dataset = get_dataset_built_in_high_level_api[1]

    try:
        artifact_path = "model"
        conda_env = os.path.join(tmpdir, "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["paddle"])

        mlflow.paddle.log_model(
            pd_model=model, artifact_path=artifact_path, conda_env=conda_env, training=True
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )

        model_retrain = paddle.Model(UCIHousing())
        optim = paddle.optimizer.Adam(learning_rate=0.015, parameters=model.parameters())
        model_retrain.prepare(optim, paddle.nn.MSELoss())
        model_retrain = mlflow.paddle.load_model(model_uri=model_uri, model=model_retrain)

        np.testing.assert_array_almost_equal(
            np.array(model.predict(test_dataset)).squeeze(),
            np.array(model_retrain.predict(test_dataset)).squeeze(),
            decimal=5,
        )
        model_path = _download_artifact_from_uri(artifact_uri=model_uri)
        model_config = Model.load(os.path.join(model_path, "MLmodel"))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
        assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
        env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]
        assert os.path.exists(os.path.join(model_path, env_path))
    finally:
        mlflow.end_run()


def test_log_model_with_pip_requirements(pd_model, tmpdir):
    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.paddle.log_model(pd_model.model, "model", pip_requirements=req_file.strpath)
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", "a"], strict=True)

    # List of requirements
    with mlflow.start_run():
        mlflow.paddle.log_model(
            pd_model.model, "model", pip_requirements=[f"-r {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.paddle.log_model(
            pd_model.model, "model", pip_requirements=[f"-c {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            ["mlflow", "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(pd_model, tmpdir):
    default_reqs = mlflow.paddle.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.paddle.log_model(pd_model.model, "model", extra_pip_requirements=req_file.strpath)
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a"])

    # List of requirements
    with mlflow.start_run():
        mlflow.paddle.log_model(
            pd_model.model, "model", extra_pip_requirements=[f"-r {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.paddle.log_model(
            pd_model.model, "model", extra_pip_requirements=[f"-c {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            ["mlflow", *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


def test_pyfunc_serve_and_score(pd_model):
    model, inference_dataframe = pd_model
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.paddle.log_model(model, artifact_path, extra_pip_requirements=[PROTOBUF_REQUIREMENT])
        model_uri = mlflow.get_artifact_uri(artifact_path)

    resp = pyfunc_serve_and_score_model(
        model_uri,
        data=pd.DataFrame(inference_dataframe),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    scores = pd.read_json(resp.content.decode("utf-8"), orient="records").values.squeeze()
    np.testing.assert_array_almost_equal(scores, model(inference_dataframe).squeeze())


def test_log_model_with_code_paths(pd_model):
    artifact_path = "model"
    with mlflow.start_run(), mock.patch(
        "mlflow.paddle._add_code_from_conf_to_system_path"
    ) as add_mock:
        mlflow.paddle.log_model(pd_model.model, artifact_path, code_paths=[__file__])
        model_uri = mlflow.get_artifact_uri(artifact_path)
        _compare_logged_code_paths(__file__, model_uri, mlflow.paddle.FLAVOR_NAME)
        mlflow.paddle.load_model(model_uri)
        add_mock.assert_called()
