import sys
import os
import pytest
from unittest import mock

import onnx
import torch
from torch import nn
import torch.onnx
from torch.utils.data import DataLoader
from sklearn import datasets
import pandas as pd
import numpy as np
import yaml

import mlflow.onnx
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature, Model
from mlflow.models.utils import _read_example
from mlflow.utils.file_utils import TempDir
from tests.helper_functions import (
    pyfunc_serve_and_score_model,
    _compare_conda_env_requirements,
    _assert_pip_requirements,
    _is_available_on_pypi,
    _compare_logged_code_paths,
)
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

TEST_DIR = "tests"
TEST_ONNX_RESOURCES_DIR = os.path.join(TEST_DIR, "resources", "onnx")

pytestmark = pytest.mark.skipif(
    (sys.version_info < (3, 6)), reason="Tests require Python 3 to run!"
)

EXTRA_PYFUNC_SERVING_TEST_ARGS = [] if _is_available_on_pypi("onnx") else ["--env-manager", "local"]


@pytest.fixture(scope="module")
def data():
    iris = datasets.load_iris()
    data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
    )
    y = data["target"]
    x = data.drop("target", axis=1)
    return x, y


@pytest.fixture(scope="module")
def dataset(data):
    x, y = data
    dataset = [(xi.astype(np.float32), yi.astype(np.float32)) for xi, yi in zip(x.values, y.values)]
    return dataset


@pytest.fixture(scope="module")
def sample_input(dataset):
    dataloader = DataLoader(dataset, batch_size=5, num_workers=1)
    # Load a batch from the data loader and return the samples
    x, _ = next(iter(dataloader))
    return x


@pytest.fixture(scope="module")
def model(dataset):
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch_size = 16
    num_workers = 4
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False
    )

    model.train()
    for _ in range(5):
        for batch in dataloader:
            optimizer.zero_grad()
            batch_size = batch[0].shape[0]
            y_pred = model(batch[0]).squeeze(dim=1)
            loss = criterion(y_pred, batch[1])
            loss.backward()
            optimizer.step()

    return model


@pytest.fixture
def onnx_model(model, sample_input, tmpdir):
    model_path = os.path.join(str(tmpdir), "torch_onnx")
    dynamic_axes = {"input": {0: "batch"}}
    torch.onnx.export(
        model, sample_input, model_path, dynamic_axes=dynamic_axes, input_names=["input"]
    )
    return onnx.load(model_path)


@pytest.fixture(scope="module")
def multi_tensor_model(dataset):
    class MyModel(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.linear = torch.nn.Linear(n, 1)
            self._train = True

        def forward(self, sepal_features, petal_features):
            if not self.training:
                if isinstance(sepal_features, np.ndarray):
                    sepal_features = torch.from_numpy(sepal_features)
                if isinstance(petal_features, np.ndarray):
                    petal_features = torch.from_numpy(petal_features)
                with torch.no_grad():
                    return self.linear(torch.cat((sepal_features, petal_features), dim=-1))
            else:
                return self.linear(torch.cat((sepal_features, petal_features), dim=-1))

    model = MyModel(4)
    model.train()
    dataloader = DataLoader(dataset, batch_size=16, num_workers=1, shuffle=True, drop_last=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(5):
        for batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(*torch.split(batch[0], 2, 1)).squeeze(dim=1)
            loss = criterion(y_pred, batch[1])
            loss.backward()
            optimizer.step()
    model.train(False)
    return model


@pytest.fixture(scope="module")
def multi_tensor_model_prediction(multi_tensor_model, data):
    x, _ = data
    feeds = {
        "sepal_features": x[x.columns[:2]].values.astype(np.float32),
        "petal_features": x[x.columns[2:4]].values.astype(np.float32),
    }
    return multi_tensor_model(**feeds).numpy().flatten()


@pytest.fixture
def multi_tensor_onnx_model(multi_tensor_model, sample_input, tmpdir):
    model_path = os.path.join(str(tmpdir), "multi_tensor_onnx")
    _sample_input = torch.split(sample_input, 2, 1)
    torch.onnx.export(
        multi_tensor_model,
        _sample_input,
        model_path,  # where to save the model (can be a file or file-like object)
        dynamic_axes={"sepal_features": [0], "petal_features": [0]},
        export_params=True,
        # store the trained parameter weights inside the model file
        do_constant_folding=True,
        # whether to execute constant folding for optimization
        input_names=["sepal_features", "petal_features"],  # the model's input names
        output_names=["target"],  # the model's output names
    )
    return onnx.load(model_path)


@pytest.fixture(scope="module")
def onnx_sklearn_model():
    """
    A scikit-learn model in ONNX format that is used to test the behavior
    of ONNX models that return outputs in list format. For reference, see
    `test_pyfunc_predict_supports_models_with_list_outputs`.
    """
    model_path = os.path.join(TEST_ONNX_RESOURCES_DIR, "sklearn_model.onnx")
    return onnx.load(model_path)


@pytest.fixture(scope="module")
def predicted(model, dataset):
    batch_size = 16
    num_workers = 4
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    predictions = np.zeros((len(dataloader.sampler),))
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            y_preds = model(batch[0]).squeeze(dim=1).numpy()
            predictions[i * batch_size : (i + 1) * batch_size] = y_preds
    return predictions


@pytest.fixture(scope="module")
def onnx_model_multiple_inputs_float64():
    model_path = os.path.join(TEST_ONNX_RESOURCES_DIR, "tf_model_multiple_inputs_float64.onnx")
    return onnx.load(model_path)


@pytest.fixture(scope="module")
def onnx_model_multiple_inputs_float32():
    model_path = os.path.join(TEST_ONNX_RESOURCES_DIR, "tf_model_multiple_inputs_float32.onnx")
    return onnx.load(model_path)


@pytest.fixture(scope="module")
def data_multiple_inputs():
    return pd.DataFrame(
        {"first_input:0": np.random.random(10), "second_input:0": np.random.random(10)}
    )


@pytest.fixture(scope="module")
def predicted_multiple_inputs(data_multiple_inputs):
    return pd.DataFrame(
        data_multiple_inputs["first_input:0"] * data_multiple_inputs["second_input:0"]
    )


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(tmpdir.strpath, "model")


@pytest.fixture
def onnx_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["onnx", "pytest", "torch"])
    return conda_env


def test_model_save_load(onnx_model, model_path):
    mlflow.onnx.save_model(onnx_model, model_path)

    # Loading ONNX model
    onnx.checker.check_model = mock.Mock()
    mlflow.onnx.load_model(model_path)
    assert onnx.checker.check_model.called


def test_signature_and_examples_are_saved_correctly(onnx_model, data, onnx_custom_env):
    model = onnx_model
    signature_ = infer_signature(*data)
    example_ = data[0].head(3)
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.onnx.save_model(
                    model,
                    path=path,
                    conda_env=onnx_custom_env,
                    signature=signature,
                    input_example=example,
                )
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


def test_model_save_load_evaluate_pyfunc_format(onnx_model, model_path, data, predicted):
    x = data[0]
    mlflow.onnx.save_model(onnx_model, model_path)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    np.testing.assert_allclose(
        pyfunc_loaded.predict(x).values.flatten(), predicted, rtol=1e-05, atol=1e-05
    )

    # pyfunc serve
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=x,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    np.testing.assert_allclose(
        pd.read_json(scoring_response.content.decode("utf-8"), orient="records")
        .values.flatten()
        .astype(np.float32),
        predicted,
        rtol=1e-05,
        atol=1e-05,
    )


def test_model_save_load_multiple_inputs(onnx_model_multiple_inputs_float64, model_path):
    mlflow.onnx.save_model(onnx_model_multiple_inputs_float64, model_path)

    # Loading ONNX model
    onnx.checker.check_model = mock.Mock()
    mlflow.onnx.load_model(model_path)
    assert onnx.checker.check_model.called


def test_model_save_load_evaluate_pyfunc_format_multiple_inputs(
    onnx_model_multiple_inputs_float64, data_multiple_inputs, predicted_multiple_inputs, model_path
):
    mlflow.onnx.save_model(onnx_model_multiple_inputs_float64, model_path)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    np.testing.assert_allclose(
        pyfunc_loaded.predict(data_multiple_inputs).values,
        predicted_multiple_inputs.values,
        rtol=1e-05,
        atol=1e-05,
    )

    # pyfunc serve
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=data_multiple_inputs,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    np.testing.assert_allclose(
        pd.read_json(scoring_response.content.decode("utf-8"), orient="records").values,
        predicted_multiple_inputs.values,
        rtol=1e-05,
        atol=1e-05,
    )


# TODO: Remove test, along with explicit casting, when https://github.com/mlflow/mlflow/issues/1286
# is fixed.


def test_pyfunc_representation_of_float32_model_casts_and_evalutes_float64_inputs(
    onnx_model_multiple_inputs_float32, model_path, data_multiple_inputs, predicted_multiple_inputs
):
    """
    The ``python_function`` representation of an MLflow model with the ONNX flavor
    casts 64-bit floats to 32-bit floats automatically before evaluating, as opposed
    to throwing an unexpected type exception. This behavior is implemented due
    to the issue described in https://github.com/mlflow/mlflow/issues/1286 where
    the JSON representation of a Pandas DataFrame does not always preserve float
    precision (e.g., 32-bit floats may be converted to 64-bit floats when persisting a
    DataFrame as JSON).
    """
    mlflow.onnx.save_model(onnx_model_multiple_inputs_float32, model_path)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    np.testing.assert_allclose(
        pyfunc_loaded.predict(data_multiple_inputs.astype("float64")).values,
        predicted_multiple_inputs.astype("float32").values,
        rtol=1e-05,
        atol=1e-05,
    )

    with pytest.raises(Exception, match="Unexpected input data type"):
        pyfunc_loaded.predict(data_multiple_inputs.astype("int32"))


def test_model_log(onnx_model):
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        try:
            if should_start_run:
                mlflow.start_run()
            artifact_path = "onnx_model"
            model_info = mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path=artifact_path)
            model_uri = "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
            assert model_info.model_uri == model_uri

            # Load model
            onnx.checker.check_model = mock.Mock()
            mlflow.onnx.load_model(model_uri)
            assert onnx.checker.check_model.called
        finally:
            mlflow.end_run()


def test_log_model_calls_register_model(onnx_model, onnx_custom_env):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path=artifact_path,
            conda_env=onnx_custom_env,
            registered_model_name="AdsModel1",
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )
        mlflow.register_model.assert_called_once_with(
            model_uri, "AdsModel1", await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )


def test_log_model_no_registered_model_name(onnx_model, onnx_custom_env):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.onnx.log_model(
            onnx_model=onnx_model, artifact_path=artifact_path, conda_env=onnx_custom_env
        )
        mlflow.register_model.assert_not_called()


def test_model_log_evaluate_pyfunc_format(onnx_model, data, predicted):
    x = data[0]

    with mlflow.start_run() as run:
        artifact_path = "onnx_model"
        mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path=artifact_path)
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=run.info.run_id, artifact_path=artifact_path
        )

        # Loading pyfunc model
        pyfunc_loaded = mlflow.pyfunc.load_model(model_uri=model_uri)
        np.testing.assert_allclose(
            pyfunc_loaded.predict(x).values.flatten(), predicted, rtol=1e-05, atol=1e-05
        )
        # test with a single numpy array
        np_ary = x.values

        # NB: Onnx wrapper returns a dictionary for non-dataframe inputs, we want to get the
        # numpy array belonging to the first (and only) model output.
        def get_ary_output(args):
            return next(iter(pyfunc_loaded.predict(args).values())).flatten()

        np.testing.assert_allclose(get_ary_output(np_ary), predicted, rtol=1e-05, atol=1e-05)
        # test with a dict with a single tensor
        np.testing.assert_allclose(
            get_ary_output({"input": np_ary}), predicted, rtol=1e-05, atol=1e-05
        )


def test_model_save_evaluate_pyfunc_format_multi_tensor(
    multi_tensor_onnx_model, data, multi_tensor_model_prediction
):
    with TempDir(chdr=True):
        path = "onnx_model"
        mlflow.onnx.save_model(onnx_model=multi_tensor_onnx_model, path=path)
        # Loading pyfunc model
        pyfunc_loaded = mlflow.pyfunc.load_model(model_uri=path)
        data, _ = data
        # get prediction
        feeds = {
            "sepal_features": data[data.columns[:2]].values,
            "petal_features": data[data.columns[2:4]].values.astype(np.float32),
        }
        preds = pyfunc_loaded.predict(feeds)["target"].flatten()
        np.testing.assert_allclose(preds, multi_tensor_model_prediction, rtol=1e-05, atol=1e-05)
        # single numpy array input should fail with the right error message:
        with pytest.raises(
            MlflowException, match="Unable to map numpy array input to the expected model input."
        ):
            pyfunc_loaded.predict(data.values)


def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    onnx_model, model_path, onnx_custom_env
):
    mlflow.onnx.save_model(onnx_model=onnx_model, path=model_path, conda_env=onnx_custom_env)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != onnx_custom_env

    with open(onnx_custom_env, "r") as f:
        onnx_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == onnx_custom_env_parsed


def test_model_save_persists_requirements_in_mlflow_model_directory(
    onnx_model, model_path, onnx_custom_env
):
    mlflow.onnx.save_model(onnx_model=onnx_model, path=model_path, conda_env=onnx_custom_env)
    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(onnx_custom_env, saved_pip_req_path)


def test_log_model_with_pip_requirements(onnx_model, tmpdir):
    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.onnx.log_model(onnx_model, "model", pip_requirements=req_file.strpath)
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", "a"], strict=True)

    # List of requirements
    with mlflow.start_run():
        mlflow.onnx.log_model(onnx_model, "model", pip_requirements=[f"-r {req_file.strpath}", "b"])
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.onnx.log_model(onnx_model, "model", pip_requirements=[f"-c {req_file.strpath}", "b"])
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            ["mlflow", "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(onnx_model, tmpdir):
    default_reqs = mlflow.onnx.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.onnx.log_model(onnx_model, "model", extra_pip_requirements=req_file.strpath)
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a"])

    # List of requirements
    with mlflow.start_run():
        mlflow.onnx.log_model(
            onnx_model, "model", extra_pip_requirements=[f"-r {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.onnx.log_model(
            onnx_model, "model", extra_pip_requirements=[f"-c {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            ["mlflow", *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


def test_model_save_accepts_conda_env_as_dict(onnx_model, model_path):
    conda_env = dict(mlflow.onnx.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.onnx.save_model(onnx_model=onnx_model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
    onnx_model, onnx_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.onnx.log_model(
            onnx_model=onnx_model, artifact_path=artifact_path, conda_env=onnx_custom_env
        )
        model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
        )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != onnx_custom_env

    with open(onnx_custom_env, "r") as f:
        onnx_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == onnx_custom_env_parsed


def test_model_log_persists_requirements_in_mlflow_model_directory(onnx_model, onnx_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.onnx.log_model(
            onnx_model=onnx_model, artifact_path=artifact_path, conda_env=onnx_custom_env
        )
        model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
        )

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(onnx_custom_env, saved_pip_req_path)


def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    onnx_model, model_path
):
    mlflow.onnx.save_model(onnx_model=onnx_model, path=model_path)
    _assert_pip_requirements(model_path, mlflow.onnx.get_default_pip_requirements())


def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    onnx_model,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path=artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)
    _assert_pip_requirements(model_uri, mlflow.onnx.get_default_pip_requirements())


def test_pyfunc_predict_supports_models_with_list_outputs(onnx_sklearn_model, model_path, data):
    """
    https://github.com/mlflow/mlflow/issues/2499
    User encountered issue where an sklearn model, converted to onnx, would return a list response.
    The issue resulted in an error because MLflow assumed it would be a numpy array. Therefore,
    the this test validates the service does not receive that error when using such a model.
    """
    x = data[0]
    mlflow.onnx.save_model(onnx_sklearn_model, model_path)
    wrapper = mlflow.pyfunc.load_model(model_path)
    wrapper.predict(pd.DataFrame(x))


def test_log_model_with_code_paths(onnx_model):
    artifact_path = "model"
    with mlflow.start_run(), mock.patch(
        "mlflow.onnx._add_code_from_conf_to_system_path"
    ) as add_mock:
        mlflow.onnx.log_model(onnx_model, artifact_path, code_paths=[__file__])
        model_uri = mlflow.get_artifact_uri(artifact_path)
        _compare_logged_code_paths(__file__, model_uri, mlflow.onnx.FLAVOR_NAME)
        mlflow.onnx.load_model(model_uri)
        add_mock.assert_called()
