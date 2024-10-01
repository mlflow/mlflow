import json
import os
from collections import namedtuple
from unittest import mock

import numpy as np
import paddle
import paddle.nn.functional as F
import pandas as pd
import pytest
import yaml
from packaging.version import Version
from paddle.nn import Linear
from sklearn import preprocessing
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

import mlflow.paddle
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.models import Model, ModelSignature
from mlflow.models.utils import _read_example, load_serving_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration

from tests.helper_functions import (
    PROTOBUF_REQUIREMENT,
    _assert_pip_requirements,
    _compare_logged_code_paths,
    _mlflow_major_version_string,
    assert_register_model_called_with_local_model_path,
    pyfunc_serve_and_score_model,
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
        def forward(self, inputs):
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


@pytest.fixture(scope="module")
def pd_model_signature():
    return ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), (-1, 10))]),
        # The _PaddleWrapper class casts numpy prediction outputs into a Pandas DataFrame.
        outputs=Schema([ColSpec(name=0, type=DataType.float)]),
    )


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


@pytest.fixture
def pd_custom_env(tmp_path):
    conda_env = os.path.join(tmp_path, "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["paddle", "pytest"])
    return conda_env


def test_model_save_load(pd_model, model_path):
    mlflow.paddle.save_model(pd_model=pd_model.model, path=model_path)

    reloaded_pd_model = mlflow.paddle.load_model(model_uri=model_path)
    reloaded_pyfunc = pyfunc.load_model(model_uri=model_path)

    np.testing.assert_array_almost_equal(
        pd_model.model(paddle.to_tensor(pd_model.inference_dataframe)),
        reloaded_pyfunc.predict(pd_model.inference_dataframe),
        decimal=5,
    )

    np.testing.assert_array_almost_equal(
        reloaded_pd_model(paddle.to_tensor(pd_model.inference_dataframe)),
        reloaded_pyfunc.predict(pd_model.inference_dataframe),
        decimal=5,
    )


def test_model_load_from_remote_uri_succeeds(pd_model, model_path, mock_s3_bucket):
    mlflow.paddle.save_model(pd_model=pd_model.model, path=model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    reloaded_model = mlflow.paddle.load_model(model_uri=model_uri)
    np.testing.assert_array_almost_equal(
        pd_model.model(paddle.to_tensor(pd_model.inference_dataframe)),
        reloaded_model(paddle.to_tensor(pd_model.inference_dataframe)),
        decimal=5,
    )


def test_model_log(pd_model, model_path, tmp_path):
    model = pd_model.model
    try:
        artifact_path = "model"
        conda_env = os.path.join(tmp_path, "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["paddle"])

        model_info = mlflow.paddle.log_model(
            pd_model=model, artifact_path=artifact_path, conda_env=conda_env
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri

        reloaded_pd_model = mlflow.paddle.load_model(model_uri=model_uri)
        np.testing.assert_array_almost_equal(
            model(paddle.to_tensor(pd_model.inference_dataframe)),
            reloaded_pd_model(paddle.to_tensor(pd_model.inference_dataframe)),
            decimal=5,
        )

        model_path = _download_artifact_from_uri(artifact_uri=model_uri)
        model_config = Model.load(os.path.join(model_path, "MLmodel"))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
        assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
        env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]["conda"]
        assert os.path.exists(os.path.join(model_path, env_path))
    finally:
        mlflow.end_run()


def test_log_model_calls_register_model(pd_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.paddle.log_model(
            pd_model=pd_model.model,
            artifact_path=artifact_path,
            registered_model_name="AdsModel1",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert_register_model_called_with_local_model_path(
            register_model_mock=mlflow.tracking._model_registry.fluent._register_model,
            model_uri=model_uri,
            registered_model_name="AdsModel1",
        )


def test_log_model_no_registered_model_name(pd_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.paddle.log_model(pd_model=pd_model.model, artifact_path=artifact_path)
        mlflow.tracking._model_registry.fluent._register_model.assert_not_called()


def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    pd_model, model_path, pd_custom_env
):
    mlflow.paddle.save_model(pd_model=pd_model.model, path=model_path, conda_env=pd_custom_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pd_custom_env

    with open(pd_custom_env) as f:
        pd_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pd_custom_env_parsed


def test_model_save_accepts_conda_env_as_dict(pd_model, model_path):
    conda_env = dict(mlflow.paddle.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.paddle.save_model(pd_model=pd_model.model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


def test_signature_and_examples_are_saved_correctly(pd_model, pd_model_signature):
    test_dataset = pd_model.inference_dataframe
    example_ = test_dataset[:3, :]
    for signature in (None, pd_model_signature):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.paddle.save_model(
                    pd_model.model, path=path, signature=signature, input_example=example
                )
                mlflow_model = Model.load(path)
                if signature is None and example is None:
                    assert mlflow_model.signature is None
                else:
                    assert mlflow_model.signature == pd_model_signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    np.testing.assert_array_equal(_read_example(mlflow_model, path), example)


def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(pd_model, pd_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.paddle.log_model(
            pd_model=pd_model.model, artifact_path=artifact_path, conda_env=pd_custom_env
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pd_custom_env

    with open(pd_custom_env) as f:
        pd_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
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

    def forward(self, inputs):
        return self.fc_(inputs)


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

    artifact_root = f"s3://{mock_s3_bucket}"
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


def test_model_built_in_high_level_api_log(pd_model_built_in_high_level_api, model_path, tmp_path):
    model = pd_model_built_in_high_level_api.model
    test_dataset = pd_model_built_in_high_level_api.inference_dataframe
    try:
        artifact_path = "model"
        conda_env = os.path.join(tmp_path, "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["paddle"])

        mlflow.paddle.log_model(pd_model=model, artifact_path=artifact_path, conda_env=conda_env)
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"

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
        env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]["conda"]
        assert os.path.exists(os.path.join(model_path, env_path))
    finally:
        mlflow.end_run()


@pytest.fixture
def model_retrain_path(tmp_path):
    return os.path.join(tmp_path, "model_retrain")


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
        match=f"Invalid object type `{error_model_type}` for `model`, must be `paddle.Model`",
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
    pd_model_built_in_high_level_api, model_path, tmp_path, get_dataset_built_in_high_level_api
):
    model = pd_model_built_in_high_level_api.model
    test_dataset = get_dataset_built_in_high_level_api[1]

    try:
        artifact_path = "model"
        conda_env = os.path.join(tmp_path, "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["paddle"])

        mlflow.paddle.log_model(
            pd_model=model, artifact_path=artifact_path, conda_env=conda_env, training=True
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"

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
        env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]["conda"]
        assert os.path.exists(os.path.join(model_path, env_path))
    finally:
        mlflow.end_run()


def test_log_model_with_pip_requirements(pd_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        mlflow.paddle.log_model(pd_model.model, "model", pip_requirements=str(req_file))
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, "a"], strict=True
        )

    # List of requirements
    with mlflow.start_run():
        mlflow.paddle.log_model(pd_model.model, "model", pip_requirements=[f"-r {req_file}", "b"])
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.paddle.log_model(pd_model.model, "model", pip_requirements=[f"-c {req_file}", "b"])
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [expected_mlflow_version, "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(pd_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    default_reqs = mlflow.paddle.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        mlflow.paddle.log_model(pd_model.model, "model", extra_pip_requirements=str(req_file))
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, *default_reqs, "a"]
        )

    # List of requirements
    with mlflow.start_run():
        mlflow.paddle.log_model(
            pd_model.model, "model", extra_pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.paddle.log_model(
            pd_model.model, "model", extra_pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [expected_mlflow_version, *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


def test_pyfunc_serve_and_score(pd_model):
    model, inference_dataframe = pd_model
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.paddle.log_model(
            model,
            artifact_path,
            extra_pip_requirements=[PROTOBUF_REQUIREMENT]
            if Version(paddle.__version__) < Version("2.5.0")
            else None,
            input_example=pd.DataFrame(inference_dataframe),
        )

    inference_payload = load_serving_example(model_info.model_uri)
    resp = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    scores = pd.DataFrame(
        data=json.loads(resp.content.decode("utf-8"))["predictions"]
    ).values.squeeze()
    np.testing.assert_array_almost_equal(
        scores, model(paddle.to_tensor(inference_dataframe)).squeeze()
    )


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


def test_model_save_load_with_metadata(pd_model, model_path):
    mlflow.paddle.save_model(
        pd_model.model, path=model_path, metadata={"metadata_key": "metadata_value"}
    )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_path)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_metadata(pd_model):
    artifact_path = "model"

    with mlflow.start_run():
        mlflow.paddle.log_model(
            pd_model.model, artifact_path=artifact_path, metadata={"metadata_key": "metadata_value"}
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_signature_inference(pd_model, pd_model_signature):
    artifact_path = "model"
    test_dataset = pd_model.inference_dataframe
    example = test_dataset[:3, :]

    with mlflow.start_run():
        mlflow.paddle.log_model(pd_model.model, artifact_path=artifact_path, input_example=example)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    mlflow_model = Model.load(model_uri)
    assert mlflow_model.signature == pd_model_signature
