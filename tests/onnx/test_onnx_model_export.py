import sys
import os
import pytest
from unittest import mock

import onnx
import torch
import torch.nn as nn
import torch.onnx
from torch.utils.data import DataLoader
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import yaml

import mlflow.onnx
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.models import infer_signature, Model
from mlflow.models.utils import _read_example
from mlflow.utils.file_utils import TempDir
from tests.helper_functions import pyfunc_serve_and_score_model
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS


TEST_DIR = "tests"
TEST_ONNX_RESOURCES_DIR = os.path.join(TEST_DIR, "resources", "onnx")

pytestmark = pytest.mark.skipif(
    (sys.version_info < (3, 6)), reason="Tests require Python 3 to run!"
)


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
    dataloader = DataLoader(dataset, batch_size=5, num_workers=1,)
    # Load a batch from the data loader and return the samples
    x, _ = next(iter(dataloader))
    return x


@pytest.fixture(scope="module")
def model(dataset):
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1),)
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
    _mlflow_conda_env(
        conda_env, additional_conda_deps=["pytest", "torch"], additional_pip_deps=["onnx"],
    )
    return conda_env


@pytest.mark.large
def test_cast_float64_to_float32():
    df = pd.DataFrame([[1.0, 2.1], [True, False]], columns=["col1", "col2"])
    df["col1"] = df["col1"].astype(np.float64)
    df["col2"] = df["col2"].astype(np.bool)
    df2 = mlflow.onnx._OnnxModelWrapper._cast_float64_to_float32(df, df.columns)
    assert df2["col1"].dtype == np.float32 and df2["col2"].dtype == np.bool


@pytest.mark.large
def test_model_save_load(onnx_model, model_path):
    mlflow.onnx.save_model(onnx_model, model_path)

    # Loading ONNX model
    onnx.checker.check_model = mock.Mock()
    mlflow.onnx.load_model(model_path)
    assert onnx.checker.check_model.called


@pytest.mark.large
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


@pytest.mark.large
def test_model_save_load_evaluate_pyfunc_format(onnx_model, model_path, data, predicted):
    x = data[0]
    mlflow.onnx.save_model(onnx_model, model_path)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_path)
    assert np.allclose(pyfunc_loaded.predict(x).values.flatten(), predicted, rtol=1e-05, atol=1e-05)

    # pyfunc serve
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=x,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    assert np.allclose(
        pd.read_json(scoring_response.content, orient="records")
        .values.flatten()
        .astype(np.float32),
        predicted,
        rtol=1e-05,
        atol=1e-05,
    )


@pytest.mark.large
def test_model_save_load_multiple_inputs(onnx_model_multiple_inputs_float64, model_path):
    mlflow.onnx.save_model(onnx_model_multiple_inputs_float64, model_path)

    # Loading ONNX model
    onnx.checker.check_model = mock.Mock()
    mlflow.onnx.load_model(model_path)
    assert onnx.checker.check_model.called


@pytest.mark.large
def test_model_save_load_evaluate_pyfunc_format_multiple_inputs(
    onnx_model_multiple_inputs_float64, data_multiple_inputs, predicted_multiple_inputs, model_path
):
    mlflow.onnx.save_model(onnx_model_multiple_inputs_float64, model_path)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_path)
    assert np.allclose(
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
    )
    assert np.allclose(
        pd.read_json(scoring_response.content, orient="records").values,
        predicted_multiple_inputs.values,
        rtol=1e-05,
        atol=1e-05,
    )


# TODO: Remove test, along with explicit casting, when https://github.com/mlflow/mlflow/issues/1286
# is fixed.
@pytest.mark.large
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
    pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_path)
    assert np.allclose(
        pyfunc_loaded.predict(data_multiple_inputs.astype("float64")).values,
        predicted_multiple_inputs.astype("float32").values,
        rtol=1e-05,
        atol=1e-05,
    )

    with pytest.raises(Exception, match="Unexpected input data type"):
        pyfunc_loaded.predict(data_multiple_inputs.astype("int32"))


@pytest.mark.large
def test_model_log(onnx_model):
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        try:
            if should_start_run:
                mlflow.start_run()
            artifact_path = "onnx_model"
            mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path=artifact_path)
            model_uri = "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )

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


@pytest.mark.large
def test_model_log_evaluate_pyfunc_format(onnx_model, data, predicted):
    x = data[0]

    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        try:
            if should_start_run:
                mlflow.start_run()
            artifact_path = "onnx_model"
            mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path=artifact_path)
            model_uri = "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )

            # Loading pyfunc model
            pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_uri=model_uri)
            assert np.allclose(
                pyfunc_loaded.predict(x).values.flatten(), predicted, rtol=1e-05, atol=1e-05
            )
        finally:
            mlflow.end_run()


@pytest.mark.large
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


@pytest.mark.large
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


@pytest.mark.large
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


@pytest.mark.large
def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    onnx_model, model_path
):
    mlflow.onnx.save_model(onnx_model=onnx_model, path=model_path, conda_env=None)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.onnx.get_default_conda_env()


@pytest.mark.large
def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    onnx_model,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path=artifact_path, conda_env=None)
        model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
        )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.onnx.get_default_conda_env()


@pytest.mark.large
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
