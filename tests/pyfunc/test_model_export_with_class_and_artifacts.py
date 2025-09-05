from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
import types
import uuid
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Dict, List
from unittest import mock

import cloudpickle
import numpy as np
import pandas as pd
import pandas.testing
import pytest
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sklearn.neighbors
import yaml

import mlflow
import mlflow.pyfunc
import mlflow.pyfunc.model
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.sklearn
from mlflow.entities import Trace
from mlflow.environment_variables import (
    MLFLOW_LOG_MODEL_COMPRESSION,
    MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING,
)
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
from mlflow.models.auth_policy import AuthPolicy, SystemAuthPolicy, UserAuthPolicy
from mlflow.models.dependencies_schemas import DependenciesSchemasType
from mlflow.models.model import _DATABRICKS_FS_LOADER_MODULE
from mlflow.models.resources import (
    DatabricksApp,
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksLakebase,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksUCConnection,
    DatabricksVectorSearchIndex,
)
from mlflow.models.utils import _read_example
from mlflow.pyfunc.context import Context, set_prediction_context
from mlflow.pyfunc.model import _load_pyfunc
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.export.inference_table import pop_trace
from mlflow.tracking.artifact_utils import (
    _download_artifact_from_uri,
)
from mlflow.types.schema import ColSpec, Map, Schema
from mlflow.types.type_hints import _infer_schema_from_list_type_hint
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.requirements_utils import _get_installed_version

import tests
from tests.helper_functions import (
    _assert_pip_requirements,
    _compare_conda_env_requirements,
    _mlflow_major_version_string,
    assert_register_model_called_with_local_model_path,
    pyfunc_serve_and_score_model,
)
from tests.tracing.helper import get_traces


def get_model_class():
    """
    Defines a custom Python model class that wraps a scikit-learn estimator.
    This can be invoked within a pytest fixture to define the class in the ``__main__`` scope.
    Alternatively, it can be invoked within a module to define the class in the module's scope.
    """

    class CustomSklearnModel(mlflow.pyfunc.PythonModel):
        def __init__(self, predict_fn):
            self.predict_fn = predict_fn

        def load_context(self, context):
            super().load_context(context)

            self.model = (
                mlflow.sklearn.load_model(model_uri=context.artifacts["sk_model"])
                if context.artifacts and "sk_model" in context.artifacts
                else None
            )

        def predict(self, context, model_input, params=None):
            return self.predict_fn(self.model, model_input)

    return CustomSklearnModel


class ModuleScopedSklearnModel(get_model_class()):
    """
    A custom Python model class defined in the test module scope.
    """


@pytest.fixture(scope="module")
def main_scoped_model_class():
    """
    A custom Python model class defined in the ``__main__`` scope.
    """
    return get_model_class()


@pytest.fixture(scope="module")
def iris_data():
    iris = sklearn.datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x, y


@pytest.fixture(scope="module")
def sklearn_knn_model(iris_data):
    x, y = iris_data
    knn_model = sklearn.neighbors.KNeighborsClassifier()
    knn_model.fit(x, y)
    return knn_model


@pytest.fixture(scope="module")
def sklearn_logreg_model(iris_data):
    x, y = iris_data
    linear_lr = sklearn.linear_model.LogisticRegression()
    linear_lr.fit(x, y)
    return linear_lr


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


@pytest.fixture
def pyfunc_custom_env(tmp_path):
    conda_env = os.path.join(tmp_path, "conda_env.yml")
    _mlflow_conda_env(
        conda_env,
        additional_pip_deps=["scikit-learn", "pytest", "cloudpickle"],
    )
    return conda_env


def _conda_env():
    # NB: We need mlflow as a dependency in the environment.
    return _mlflow_conda_env(
        additional_pip_deps=[
            f"cloudpickle=={cloudpickle.__version__}",
            f"scikit-learn=={sklearn.__version__}",
        ],
    )


def test_model_save_load(sklearn_knn_model, main_scoped_model_class, iris_data, tmp_path):
    sklearn_model_path = os.path.join(tmp_path, "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")

    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        conda_env=_conda_env(),
        python_model=main_scoped_model_class(test_predict),
    )

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_path)
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict(iris_data[0]),
        test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]),
    )


@pytest.mark.skip(
    reason="In MLflow 3.0, `log_model` does not start a run. Consider removing this test."
)
def test_pyfunc_model_log_load_no_active_run(sklearn_knn_model, main_scoped_model_class, iris_data):
    sklearn_artifact_path = "sk_model_no_run"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sklearn_knn_model, name=sklearn_artifact_path)
        sklearn_model_uri = f"runs:/{mlflow.active_run().info.run_id}/{sklearn_artifact_path}"

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_artifact_path = "pyfunc_model"
    assert mlflow.active_run() is None
    mlflow.pyfunc.log_model(
        name=pyfunc_artifact_path,
        artifacts={"sk_model": sklearn_model_uri},
        python_model=main_scoped_model_class(test_predict),
    )
    pyfunc_model_uri = f"runs:/{mlflow.active_run().info.run_id}/{pyfunc_artifact_path}"
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_uri)
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict(iris_data[0]),
        test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]),
    )
    mlflow.end_run()


def test_model_log_load(sklearn_knn_model, main_scoped_model_class, iris_data):
    sklearn_artifact_path = "sk_model"
    with mlflow.start_run():
        sklearn_model_info = mlflow.sklearn.log_model(sklearn_knn_model, name=sklearn_artifact_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        pyfunc_model_info = mlflow.pyfunc.log_model(
            name=pyfunc_artifact_path,
            artifacts={"sk_model": sklearn_model_info.model_uri},
            python_model=main_scoped_model_class(test_predict),
        )
        pyfunc_model_path = _download_artifact_from_uri(pyfunc_model_info.model_uri)
        model_config = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_info.model_uri)
    assert model_config.to_yaml() == loaded_pyfunc_model.metadata.to_yaml()
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict(iris_data[0]),
        test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]),
    )


def test_python_model_predict_compatible_without_params(sklearn_knn_model, iris_data):
    class CustomSklearnModelWithoutParams(mlflow.pyfunc.PythonModel):
        def __init__(self, predict_fn):
            self.predict_fn = predict_fn

        def load_context(self, context):
            super().load_context(context)

            self.model = mlflow.sklearn.load_model(model_uri=context.artifacts["sk_model"])

        def predict(self, context, model_input):
            return self.predict_fn(self.model, model_input)

    sklearn_artifact_path = "sk_model"
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(sklearn_knn_model, name=sklearn_artifact_path)
        sklearn_model_uri = model_info.model_uri

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name=pyfunc_artifact_path,
            artifacts={"sk_model": sklearn_model_uri},
            python_model=CustomSklearnModelWithoutParams(test_predict),
        )
        pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri)
        model_config = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    assert model_config.to_yaml() == loaded_pyfunc_model.metadata.to_yaml()
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict(iris_data[0]),
        test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]),
    )


def test_signature_and_examples_are_saved_correctly(iris_data, main_scoped_model_class, tmp_path):
    sklearn_model_path = str(tmp_path.joinpath("sklearn_model"))
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    data = iris_data
    signature_ = infer_signature(*data)
    example_ = data[0][:3]
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.pyfunc.save_model(
                    path=path,
                    artifacts={"sk_model": sklearn_model_path},
                    python_model=main_scoped_model_class(test_predict),
                    signature=signature,
                    input_example=example,
                )
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    np.testing.assert_array_equal(_read_example(mlflow_model, path), example)


class DummyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


def test_log_model_calls_register_model(sklearn_knn_model, main_scoped_model_class):
    with mlflow.start_run():
        with mock.patch(
            "mlflow.tracking._model_registry.fluent._register_model"
        ) as register_model_mock:
            registered_model_name = "AdsModel1"
            pyfunc_model_info = mlflow.pyfunc.log_model(
                name="pyfunc_model",
                python_model=DummyModel(),
                registered_model_name=registered_model_name,
            )
        assert_register_model_called_with_local_model_path(
            register_model_mock, pyfunc_model_info.model_uri, registered_model_name
        )


def test_log_model_no_registered_model_name(sklearn_knn_model, main_scoped_model_class):
    with mlflow.start_run():
        with mock.patch(
            "mlflow.tracking._model_registry.fluent._register_model"
        ) as register_model_mock:
            mlflow.pyfunc.log_model(
                name="pyfunc_model",
                python_model=DummyModel(),
            )
        register_model_mock.assert_not_called()


def test_model_load_from_remote_uri_succeeds(
    sklearn_knn_model, main_scoped_model_class, tmp_path, mock_s3_bucket, iris_data
):
    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_repo = S3ArtifactRepository(artifact_root)

    sklearn_model_path = os.path.join(tmp_path, "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)
    sklearn_artifact_path = "sk_model"
    artifact_repo.log_artifacts(sklearn_model_path, artifact_path=sklearn_artifact_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(test_predict),
        conda_env=_conda_env(),
    )

    pyfunc_artifact_path = "pyfunc_model"
    artifact_repo.log_artifacts(pyfunc_model_path, artifact_path=pyfunc_artifact_path)

    model_uri = artifact_root + "/" + pyfunc_artifact_path
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict(iris_data[0]),
        test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]),
    )


def test_add_to_model_adds_specified_kwargs_to_mlmodel_configuration():
    custom_kwargs = {
        "key1": "value1",
        "key2": 20,
        "key3": range(10),
    }
    model_config = Model()
    mlflow.pyfunc.add_to_model(
        model=model_config,
        loader_module=os.path.basename(__file__)[:-3],
        data="data",
        code="code",
        env=None,
        **custom_kwargs,
    )

    assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
    assert all(item in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME] for item in custom_kwargs)


def test_pyfunc_model_serving_without_conda_env_activation_succeeds_with_main_scoped_class(
    sklearn_knn_model, main_scoped_model_class, iris_data, tmp_path
):
    sklearn_model_path = os.path.join(tmp_path, "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(test_predict),
        conda_env=_conda_env(),
    )
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_path)

    sample_input = pd.DataFrame(iris_data[0])
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=pyfunc_model_path,
        data=sample_input,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert scoring_response.status_code == 200
    np.testing.assert_array_equal(
        np.array(json.loads(scoring_response.text)["predictions"]),
        loaded_pyfunc_model.predict(sample_input),
    )


def test_pyfunc_model_serving_with_conda_env_activation_succeeds_with_main_scoped_class(
    sklearn_knn_model, main_scoped_model_class, iris_data, tmp_path
):
    sklearn_model_path = os.path.join(tmp_path, "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(test_predict),
        conda_env=_conda_env(),
    )
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_path)

    sample_input = pd.DataFrame(iris_data[0])
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=pyfunc_model_path,
        data=sample_input,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    assert scoring_response.status_code == 200
    np.testing.assert_array_equal(
        np.array(json.loads(scoring_response.text)["predictions"]),
        loaded_pyfunc_model.predict(sample_input),
    )


def test_pyfunc_model_serving_without_conda_env_activation_succeeds_with_module_scoped_class(
    sklearn_knn_model, iris_data, tmp_path
):
    sklearn_model_path = os.path.join(tmp_path, "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=ModuleScopedSklearnModel(test_predict),
        code_paths=[os.path.dirname(tests.__file__)],
        conda_env=_conda_env(),
    )
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_path)

    sample_input = pd.DataFrame(iris_data[0])
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=pyfunc_model_path,
        data=sample_input,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert scoring_response.status_code == 200
    np.testing.assert_array_equal(
        np.array(json.loads(scoring_response.text)["predictions"]),
        loaded_pyfunc_model.predict(sample_input),
    )


def test_pyfunc_cli_predict_command_without_conda_env_activation_succeeds(
    sklearn_knn_model, main_scoped_model_class, iris_data, tmp_path
):
    sklearn_model_path = os.path.join(tmp_path, "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(test_predict),
        conda_env=_conda_env(),
    )
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_path)

    sample_input = pd.DataFrame(iris_data[0])
    input_csv_path = os.path.join(tmp_path, "input with spaces.csv")
    sample_input.to_csv(input_csv_path, header=True, index=False)
    output_json_path = os.path.join(tmp_path, "output.json")
    process = Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "--model-uri",
            pyfunc_model_path,
            "-i",
            input_csv_path,
            "--content-type",
            "csv",
            "-o",
            output_json_path,
            "--env-manager",
            "local",
        ],
        stdout=PIPE,
        stderr=PIPE,
        preexec_fn=os.setsid,
    )
    _, stderr = process.communicate()
    assert process.wait() == 0, f"stderr = \n\n{stderr}\n\n"
    with open(output_json_path) as f:
        result_df = pd.DataFrame(data=json.load(f)["predictions"])
    np.testing.assert_array_equal(
        result_df.values.transpose()[0], loaded_pyfunc_model.predict(sample_input)
    )


def test_pyfunc_cli_predict_command_with_conda_env_activation_succeeds(
    sklearn_knn_model, main_scoped_model_class, iris_data, tmp_path
):
    sklearn_model_path = os.path.join(tmp_path, "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(test_predict),
        conda_env=_conda_env(),
    )
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_path)

    sample_input = pd.DataFrame(iris_data[0])
    input_csv_path = os.path.join(tmp_path, "input with spaces.csv")
    sample_input.to_csv(input_csv_path, header=True, index=False)
    output_json_path = os.path.join(tmp_path, "output.json")
    process = Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "--model-uri",
            pyfunc_model_path,
            "-i",
            input_csv_path,
            "--content-type",
            "csv",
            "-o",
            output_json_path,
        ],
        stderr=PIPE,
        stdout=PIPE,
        preexec_fn=os.setsid,
    )
    stdout, stderr = process.communicate()
    assert process.wait() == 0, f"stdout = \n\n{stdout}\n\n stderr = \n\n{stderr}\n\n"
    with open(output_json_path) as f:
        result_df = pandas.DataFrame(json.load(f)["predictions"])
    np.testing.assert_array_equal(
        result_df.values.transpose()[0], loaded_pyfunc_model.predict(sample_input)
    )


def test_save_model_persists_specified_conda_env_in_mlflow_model_directory(
    sklearn_knn_model, main_scoped_model_class, pyfunc_custom_env, tmp_path
):
    sklearn_model_path = os.path.join(tmp_path, "sklearn_model")
    mlflow.sklearn.save_model(
        sk_model=sklearn_knn_model,
        path=sklearn_model_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(predict_fn=None),
        conda_env=pyfunc_custom_env,
    )

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )
    saved_conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pyfunc_custom_env

    with open(pyfunc_custom_env) as f:
        pyfunc_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pyfunc_custom_env_parsed


def test_save_model_persists_requirements_in_mlflow_model_directory(
    sklearn_knn_model, main_scoped_model_class, pyfunc_custom_env, tmp_path
):
    sklearn_model_path = os.path.join(tmp_path, "sklearn_model")
    mlflow.sklearn.save_model(
        sk_model=sklearn_knn_model,
        path=sklearn_model_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(predict_fn=None),
        conda_env=pyfunc_custom_env,
    )

    saved_pip_req_path = os.path.join(pyfunc_model_path, "requirements.txt")
    _compare_conda_env_requirements(pyfunc_custom_env, saved_pip_req_path)


def test_log_model_with_pip_requirements(sklearn_knn_model, main_scoped_model_class, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    python_model = main_scoped_model_class(predict_fn=None)
    sklearn_model_path = os.path.join(tmp_path, "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)
    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=python_model,
            pip_requirements=str(req_file),
            artifacts={"sk_model": sklearn_model_path},
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, "a"],
            strict=True,
        )

    # List of requirements
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=python_model,
            pip_requirements=[f"-r {req_file}", "b"],
            artifacts={"sk_model": sklearn_model_path},
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, "a", "b"],
            strict=True,
        )

    # Constraints file
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=python_model,
            pip_requirements=[f"-c {req_file}", "b"],
            artifacts={"sk_model": sklearn_model_path},
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(
    sklearn_knn_model, main_scoped_model_class, tmp_path
):
    expected_mlflow_version = _mlflow_major_version_string()
    sklearn_model_path = str(tmp_path.joinpath("sklearn_model"))
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    python_model = main_scoped_model_class(predict_fn=None)
    default_reqs = mlflow.pyfunc.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=python_model,
            artifacts={"sk_model": sklearn_model_path},
            extra_pip_requirements=str(req_file),
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, *default_reqs, "a"],
        )

    # List of requirements
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            artifacts={"sk_model": sklearn_model_path},
            python_model=python_model,
            extra_pip_requirements=[f"-r {req_file}", "b"],
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, *default_reqs, "a", "b"],
        )

    # Constraints file
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            artifacts={"sk_model": sklearn_model_path},
            python_model=python_model,
            extra_pip_requirements=[f"-c {req_file}", "b"],
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


def test_log_model_persists_specified_conda_env_in_mlflow_model_directory(
    sklearn_knn_model, main_scoped_model_class, pyfunc_custom_env
):
    sklearn_artifact_path = "sk_model"
    with mlflow.start_run():
        sklearn_model_info = mlflow.sklearn.log_model(sklearn_knn_model, name=sklearn_artifact_path)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        pyfunc_model_info = mlflow.pyfunc.log_model(
            name=pyfunc_artifact_path,
            artifacts={"sk_model": sklearn_model_info.model_uri},
            python_model=main_scoped_model_class(predict_fn=None),
            conda_env=pyfunc_custom_env,
        )
        pyfunc_model_path = _download_artifact_from_uri(pyfunc_model_info.model_uri)

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )
    saved_conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pyfunc_custom_env

    with open(pyfunc_custom_env) as f:
        pyfunc_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pyfunc_custom_env_parsed


def test_model_log_persists_requirements_in_mlflow_model_directory(
    sklearn_knn_model, main_scoped_model_class, pyfunc_custom_env
):
    sklearn_artifact_path = "sk_model"
    with mlflow.start_run():
        sklearn_model_info = mlflow.sklearn.log_model(sklearn_knn_model, name=sklearn_artifact_path)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        pyfunc_model_info = mlflow.pyfunc.log_model(
            name=pyfunc_artifact_path,
            artifacts={"sk_model": sklearn_model_info.model_uri},
            python_model=main_scoped_model_class(predict_fn=None),
            conda_env=pyfunc_custom_env,
        )
        pyfunc_model_path = _download_artifact_from_uri(pyfunc_model_info.model_uri)

    saved_pip_req_path = os.path.join(pyfunc_model_path, "requirements.txt")
    _compare_conda_env_requirements(pyfunc_custom_env, saved_pip_req_path)


def test_save_model_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    sklearn_logreg_model, main_scoped_model_class, tmp_path
):
    sklearn_model_path = os.path.join(tmp_path, "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_logreg_model, path=sklearn_model_path)

    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(predict_fn=None),
        conda_env=_conda_env(),
    )
    _assert_pip_requirements(pyfunc_model_path, mlflow.pyfunc.get_default_pip_requirements())


def test_log_model_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    sklearn_knn_model, main_scoped_model_class
):
    sklearn_artifact_path = "sk_model"
    with mlflow.start_run():
        sklearn_model_info = mlflow.sklearn.log_model(sklearn_knn_model, name=sklearn_artifact_path)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        pyfunc_model_info = mlflow.pyfunc.log_model(
            name=pyfunc_artifact_path,
            artifacts={
                "sk_model": sklearn_model_info.model_uri,
            },
            python_model=main_scoped_model_class(predict_fn=None),
        )
    _assert_pip_requirements(
        pyfunc_model_info.model_uri, mlflow.pyfunc.get_default_pip_requirements()
    )


def test_save_model_correctly_resolves_directory_artifact_with_nested_contents(
    tmp_path, model_path, iris_data
):
    directory_artifact_path = os.path.join(tmp_path, "directory_artifact")
    nested_file_relative_path = os.path.join(
        "my", "somewhat", "heavily", "nested", "directory", "myfile.txt"
    )
    nested_file_path = os.path.join(directory_artifact_path, nested_file_relative_path)
    os.makedirs(os.path.dirname(nested_file_path))
    nested_file_text = "some sample file text"
    with open(nested_file_path, "w") as f:
        f.write(nested_file_text)

    class ArtifactValidationModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            expected_file_path = os.path.join(
                context.artifacts["testdir"], nested_file_relative_path
            )
            if not os.path.exists(expected_file_path):
                return False
            else:
                with open(expected_file_path) as f:
                    return f.read() == nested_file_text

    mlflow.pyfunc.save_model(
        path=model_path,
        artifacts={"testdir": directory_artifact_path},
        python_model=ArtifactValidationModel(),
        conda_env=_conda_env(),
    )

    loaded_model = mlflow.pyfunc.load_model(model_uri=model_path)
    assert loaded_model.predict(iris_data[0])


def test_save_model_with_no_artifacts_does_not_produce_artifacts_dir(model_path):
    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=ModuleScopedSklearnModel(predict_fn=None),
        artifacts=None,
        conda_env=_conda_env(),
    )

    assert os.path.exists(model_path)
    assert "artifacts" not in os.listdir(model_path)
    pyfunc_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )
    assert mlflow.pyfunc.model.CONFIG_KEY_ARTIFACTS not in pyfunc_conf


def test_save_model_with_python_model_argument_of_invalid_type_raises_exception(
    tmp_path,
):
    with pytest.raises(
        MlflowException,
        match="must be a PythonModel instance, callable object, or path to a",
    ):
        mlflow.pyfunc.save_model(path=os.path.join(tmp_path, "model1"), python_model=5)

    with pytest.raises(
        MlflowException,
        match="must be a PythonModel instance, callable object, or path to a",
    ):
        mlflow.pyfunc.save_model(
            path=os.path.join(tmp_path, "model2"), python_model=["not a python model"]
        )
    with pytest.raises(MlflowException, match="The provided model path"):
        mlflow.pyfunc.save_model(
            path=os.path.join(tmp_path, "model3"), python_model="not a valid filepath"
        )


def test_save_model_with_unsupported_argument_combinations_throws_exception(model_path):
    with pytest.raises(
        MlflowException,
        match="Either `loader_module` or `python_model` must be specified",
    ) as exc_info:
        mlflow.pyfunc.save_model(
            path=model_path,
            artifacts={"artifact": "/path/to/artifact"},
            python_model=None,
        )

    python_model = ModuleScopedSklearnModel(predict_fn=None)
    loader_module = __name__
    with pytest.raises(
        MlflowException,
        match="The following sets of parameters cannot be specified together",
    ) as exc_info:
        mlflow.pyfunc.save_model(
            path=model_path, python_model=python_model, loader_module=loader_module
        )
    assert str(python_model) in str(exc_info)
    assert str(loader_module) in str(exc_info)

    with pytest.raises(
        MlflowException,
        match="The following sets of parameters cannot be specified together",
    ) as exc_info:
        mlflow.pyfunc.save_model(
            path=model_path,
            python_model=python_model,
            data_path="/path/to/data",
            artifacts={"artifact": "/path/to/artifact"},
        )

    with pytest.raises(
        MlflowException,
        match="Either `loader_module` or `python_model` must be specified",
    ):
        mlflow.pyfunc.save_model(path=model_path, python_model=None, loader_module=None)


def test_log_model_with_unsupported_argument_combinations_throws_exception():
    match = (
        "Either `loader_module` or `python_model` must be specified. A `loader_module` "
        "should be a python module. A `python_model` should be a subclass of "
        "PythonModel"
    )
    with mlflow.start_run(), pytest.raises(MlflowException, match=match):
        mlflow.pyfunc.log_model(
            name="pyfunc_model",
            artifacts={"artifact": "/path/to/artifact"},
            python_model=None,
        )

    python_model = ModuleScopedSklearnModel(predict_fn=None)
    loader_module = __name__
    with (
        mlflow.start_run(),
        pytest.raises(
            MlflowException,
            match="The following sets of parameters cannot be specified together",
        ) as exc_info,
    ):
        mlflow.pyfunc.log_model(
            name="pyfunc_model",
            python_model=python_model,
            loader_module=loader_module,
        )
    assert str(python_model) in str(exc_info)
    assert str(loader_module) in str(exc_info)

    with (
        mlflow.start_run(),
        pytest.raises(
            MlflowException,
            match="The following sets of parameters cannot be specified together",
        ) as exc_info,
    ):
        mlflow.pyfunc.log_model(
            name="pyfunc_model",
            python_model=python_model,
            data_path="/path/to/data",
            artifacts={"artifact1": "/path/to/artifact"},
        )

    with (
        mlflow.start_run(),
        pytest.raises(
            MlflowException,
            match="Either `loader_module` or `python_model` must be specified",
        ),
    ):
        mlflow.pyfunc.log_model(name="pyfunc_model", python_model=None, loader_module=None)


def test_repr_can_be_called_without_run_id_or_artifact_path():
    model_meta = Model(
        artifact_path=None,
        run_id=None,
        flavors={"python_function": {"loader_module": "someFlavour"}},
    )

    class TestModel:
        def predict(self, model_input, params=None):
            return model_input

    model_impl = TestModel()

    assert "flavor: someFlavour" in mlflow.pyfunc.PyFuncModel(model_meta, model_impl).__repr__()


def test_load_model_with_differing_cloudpickle_version_at_micro_granularity_logs_warning(
    model_path,
):
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    mlflow.pyfunc.save_model(path=model_path, python_model=TestModel())
    saver_cloudpickle_version = "0.5.8"
    model_config_path = os.path.join(model_path, "MLmodel")
    model_config = Model.load(model_config_path)
    model_config.flavors[mlflow.pyfunc.FLAVOR_NAME][
        mlflow.pyfunc.model.CONFIG_KEY_CLOUDPICKLE_VERSION
    ] = saver_cloudpickle_version
    model_config.save(model_config_path)

    log_messages = []

    def custom_warn(message_text, *args, **kwargs):
        log_messages.append(message_text % args % kwargs)

    loader_cloudpickle_version = "0.5.7"
    with (
        mock.patch("mlflow.pyfunc._logger.warning") as warn_mock,
        mock.patch("cloudpickle.__version__") as cloudpickle_version_mock,
    ):
        cloudpickle_version_mock.__str__ = lambda *args, **kwargs: loader_cloudpickle_version
        warn_mock.side_effect = custom_warn
        mlflow.pyfunc.load_model(model_uri=model_path)

    assert any(
        "differs from the version of CloudPickle that is currently running" in log_message
        and saver_cloudpickle_version in log_message
        and loader_cloudpickle_version in log_message
        for log_message in log_messages
    )


def test_load_model_with_missing_cloudpickle_version_logs_warning(model_path):
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    mlflow.pyfunc.save_model(path=model_path, python_model=TestModel())
    model_config_path = os.path.join(model_path, "MLmodel")
    model_config = Model.load(model_config_path)
    del model_config.flavors[mlflow.pyfunc.FLAVOR_NAME][
        mlflow.pyfunc.model.CONFIG_KEY_CLOUDPICKLE_VERSION
    ]
    model_config.save(model_config_path)

    log_messages = []

    def custom_warn(message_text, *args, **kwargs):
        log_messages.append(message_text % args % kwargs)

    with mock.patch("mlflow.pyfunc._logger.warning") as warn_mock:
        warn_mock.side_effect = custom_warn
        mlflow.pyfunc.load_model(model_uri=model_path)

    assert any(
        (
            "The version of CloudPickle used to save the model could not be found"
            " in the MLmodel configuration"
        )
        in log_message
        for log_message in log_messages
    )


def test_save_and_load_model_with_special_chars(
    sklearn_knn_model, main_scoped_model_class, iris_data, tmp_path
):
    sklearn_model_path = os.path.join(tmp_path, "sklearn_  model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    # Intentionally create a path that has non-url-compatible characters
    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_ :% model")

    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        conda_env=_conda_env(),
        python_model=main_scoped_model_class(test_predict),
    )

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_path)
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict(iris_data[0]),
        test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]),
    )


def test_model_with_code_path_containing_main(tmp_path):
    """Test that the __main__ module is unaffected by model loading"""
    directory = tmp_path.joinpath("model_with_main")
    directory.mkdir()
    main = directory.joinpath("__main__.py")
    main.write_text("# empty main")
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=mlflow.pyfunc.model.PythonModel(),
            code_paths=[str(directory)],
        )

    assert "__main__" in sys.modules
    mlflow.pyfunc.load_model(model_info.model_uri)
    assert "__main__" in sys.modules


def test_model_save_load_with_metadata(tmp_path):
    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")

    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        conda_env=_conda_env(),
        python_model=mlflow.pyfunc.model.PythonModel(),
        metadata={"metadata_key": "metadata_value"},
    )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_path)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_metadata():
    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            name=pyfunc_artifact_path,
            python_model=mlflow.pyfunc.model.PythonModel(),
            metadata={"metadata_key": "metadata_value"},
        )
        pyfunc_model_uri = f"runs:/{mlflow.active_run().info.run_id}/{pyfunc_artifact_path}"

    reloaded_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


class SklearnModel(mlflow.pyfunc.PythonModel):
    def __init__(self) -> None:
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()

    def predict(self, context, model_input, params=None):
        return self.model.predict(model_input)


def test_dependency_inference_does_not_exclude_mlflow_dependencies(tmp_path):
    mlflow.pyfunc.save_model(
        path=tmp_path,
        python_model=SklearnModel(),
    )
    requiments = tmp_path.joinpath("requirements.txt").read_text()
    assert f"scikit-learn=={sklearn.__version__}" in requiments


def test_functional_python_model_no_type_hints(tmp_path):
    def python_model(x):
        return x

    mlflow.pyfunc.save_model(path=tmp_path, python_model=python_model, input_example=[{"a": "b"}])
    model = Model.load(tmp_path)
    assert model.signature.inputs == Schema([ColSpec("string", name="a")])
    assert model.signature.outputs == Schema([ColSpec("string", name="a")])


def list_to_list(x: List[str]) -> List[str]:  # noqa: UP006
    return x


def list_dict_to_list(x: List[Dict[str, str]]) -> List[str]:  # noqa: UP006
    return ["".join((*d.keys(), *d.values())) for d in x]  # join keys and values


def test_functional_python_model_list_dict_to_list_without_example(tmp_path):
    mlflow.pyfunc.save_model(
        path=tmp_path, python_model=list_dict_to_list, pip_requirements=["pandas"]
    )
    model = Model.load(tmp_path)
    assert model.signature.inputs == Schema([ColSpec(Map("string"))])
    assert model.signature.outputs == Schema([ColSpec("string")])
    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    assert loaded_model.predict([{"a": "x"}, {"a": "y"}]) == ["ax", "ay"]


@pytest.mark.parametrize(
    ("input_example"),
    [
        [0],
        [{"a": "b"}],
    ],
)
def test_functional_python_model_list_invalid_example(tmp_path, input_example):
    with mock.patch("mlflow.models.signature._logger.warning") as mock_warning:
        mlflow.pyfunc.save_model(
            path=tmp_path, python_model=list_to_list, input_example=input_example
        )
        assert any(
            "Input example is not compatible with the type hint" in call[0][0]
            for call in mock_warning.call_args_list
        )


@pytest.mark.parametrize(
    "input_example",
    [
        ["a"],
        [{0: "a"}],
        [{"a": 0}],
    ],
)
def test_functional_python_model_list_dict_invalid_example(tmp_path, input_example):
    with mock.patch("mlflow.models.signature._logger.warning") as mock_warning:
        mlflow.pyfunc.save_model(
            path=tmp_path, python_model=list_dict_to_list, input_example=input_example
        )
        assert any(
            "Input example is not compatible with the type hint" in call[0][0]
            for call in mock_warning.call_args_list
        )


def test_functional_python_model_list_dict_to_list(tmp_path):
    mlflow.pyfunc.save_model(
        path=tmp_path,
        python_model=list_dict_to_list,
        input_example=[{"a": "x", "b": "y"}],
    )
    model = Model.load(tmp_path)
    assert model.signature.inputs == Schema([ColSpec(Map("string"))])
    assert model.signature.outputs == Schema([ColSpec("string")])
    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    assert loaded_model.predict([{"a": "x", "b": "y"}]) == ["abxy"]


def list_dict_to_list_dict(x: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{v: k for k, v in d.items()} for d in x]  # swap keys and values


def test_functional_python_model_list_dict_to_list_dict():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="test_model",
            python_model=list_dict_to_list_dict,
            input_example=[{"a": "x", "b": "y"}],
        )

    assert model_info.signature.inputs.to_dict() == [
        {"type": "map", "values": {"type": "string"}, "required": True}
    ]
    assert model_info.signature.outputs.to_dict() == [
        {"type": "map", "values": {"type": "string"}, "required": True}
    ]

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_model.predict([{"a": "x", "b": "y"}]) == [{"x": "a", "y": "b"}]


def test_list_dict_with_signature_override():
    class CustomModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input: list[dict[str, str]], params=None):
            return model_input

    signature = infer_signature([{"a": "x", "b": "y"}, {"a": "z"}])
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="test_model",
            python_model=CustomModel(),
            signature=signature,
        )
    assert model_info.signature.inputs == _infer_schema_from_list_type_hint(list[dict[str, str]])
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_model.predict([{"a": "z"}]) == [{"a": "z"}]


def list_dict_to_list_dict_pep585(x: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{v: k for k, v in d.items()} for d in x]  # swap keys and values


def test_functional_python_model_list_dict_to_list_dict_with_example_pep585(tmp_path):
    mlflow.pyfunc.save_model(
        path=tmp_path,
        python_model=list_dict_to_list_dict_pep585,
        input_example=[{"a": "x", "b": "y"}],
    )
    model = Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"type": "map", "values": {"type": "string"}, "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "map", "values": {"type": "string"}, "required": True},
    ]
    loaded_model = mlflow.pyfunc.load_model(tmp_path)
    assert loaded_model.predict([{"a": "x", "b": "y"}]) == [{"x": "a", "y": "b"}]


def multiple_arguments(x: list[str], y: list[str]) -> list[str]:
    return x + y


def test_functional_python_model_multiple_arguments(tmp_path):
    with pytest.raises(
        MlflowException, match=r"must accept exactly one argument\. Found 2 arguments\."
    ):
        mlflow.pyfunc.save_model(path=tmp_path, python_model=multiple_arguments)


def no_arguments() -> list[str]:
    return []


def test_functional_python_model_no_arguments(tmp_path):
    with pytest.raises(
        MlflowException, match=r"must accept exactly one argument\. Found 0 arguments\."
    ):
        mlflow.pyfunc.save_model(path=tmp_path, python_model=no_arguments)


def requires_sklearn(x: list[str]) -> list[str]:
    import sklearn  # noqa: F401

    return x


def test_functional_python_model_infer_requirements(tmp_path):
    mlflow.pyfunc.save_model(path=tmp_path, python_model=requires_sklearn, input_example=["a"])
    assert "scikit-learn==" in tmp_path.joinpath("requirements.txt").read_text()


def test_functional_python_model_throws_when_required_arguments_are_missing(tmp_path):
    mlflow.pyfunc.save_model(
        path=tmp_path / uuid.uuid4().hex,
        python_model=requires_sklearn,
        input_example=["a"],
    )
    mlflow.pyfunc.save_model(
        path=tmp_path / uuid.uuid4().hex,
        python_model=requires_sklearn,
        pip_requirements=["scikit-learn"],
    )
    mlflow.pyfunc.save_model(
        path=tmp_path / uuid.uuid4().hex,
        python_model=requires_sklearn,
        extra_pip_requirements=["scikit-learn"],
    )
    with pytest.raises(MlflowException, match="at least one of"):
        mlflow.pyfunc.save_model(path=tmp_path / uuid.uuid4().hex, python_model=requires_sklearn)


class AnnotatedPythonModel(mlflow.pyfunc.PythonModel):
    def predict(self, context: dict[str, Any], model_input: list[str], params=None) -> list[str]:
        assert isinstance(model_input, list)
        assert all(isinstance(x, str) for x in model_input)
        return model_input


def test_class_python_model_type_hints(tmp_path):
    mlflow.pyfunc.save_model(path=tmp_path, python_model=AnnotatedPythonModel())
    model = Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [{"type": "string", "required": True}]
    assert model.signature.outputs.to_dict() == [{"type": "string", "required": True}]
    model = mlflow.pyfunc.load_model(tmp_path)
    assert model.predict(["a", "b"]) == ["a", "b"]


def test_python_model_predict_with_params():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="test_model",
            python_model=AnnotatedPythonModel(),
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert loaded_model.predict(["a", "b"], params={"foo": [0, 1]}) == ["a", "b"]
    assert loaded_model.predict(["a", "b"], params={"foo": np.array([0, 1])}) == [
        "a",
        "b",
    ]


def test_python_model_with_type_hint_errors_with_different_signature():
    signature = infer_signature(["input1", "input2"], params={"foo": [8]})

    with mlflow.start_run():
        with mock.patch("mlflow.pyfunc._logger.warning") as warn_mock:
            mlflow.pyfunc.log_model(
                name="test_model",
                python_model=AnnotatedPythonModel(),
                signature=signature,
            )
        warn_mock.assert_called_once()
        assert (
            "Provided signature does not match the signature inferred from"
            in warn_mock.call_args[0][0]
        )


def test_artifact_path_posix(sklearn_knn_model, main_scoped_model_class, tmp_path):
    sklearn_model_path = tmp_path.joinpath("sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = tmp_path.joinpath("pyfunc_model")

    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": str(sklearn_model_path)},
        conda_env=_conda_env(),
        python_model=main_scoped_model_class(test_predict),
    )

    artifacts = _load_pyfunc(pyfunc_model_path).context.artifacts
    assert all("\\" not in artifact_uri for artifact_uri in artifacts.values())


def test_load_model_fails_for_feature_store_models(tmp_path):
    feature_store = os.path.join(tmp_path, "feature_store")
    os.mkdir(feature_store)
    feature_spec = os.path.join(feature_store, "feature_spec.yaml")
    with open(feature_spec, "w+") as f:
        f.write("contents")

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name="model",
            data_path=feature_store,
            loader_module=_DATABRICKS_FS_LOADER_MODULE,
            code_paths=[__file__],
        )
    with pytest.raises(
        MlflowException,
        match="Note: mlflow.pyfunc.load_model is not supported for Feature Store models",
    ):
        mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")


def test_pyfunc_model_infer_signature_from_type_hints():
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input: list[str], params=None) -> list[str]:
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="test_model",
            python_model=TestModel(),
            input_example=["a"],
        )
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_model.metadata.get_input_schema() == Schema([ColSpec("string")])
    assert pyfunc_model.predict(["a", "b"]) == ["a", "b"]


def test_streamable_model_save_load(iris_data, tmp_path):
    class StreamableModel(mlflow.pyfunc.PythonModel):
        def __init__(self):
            pass

        def predict(self, context, model_input, params=None):
            pass

        def predict_stream(self, context, model_input, params=None):
            yield "test1"
            yield "test2"

    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")

    python_model = StreamableModel()

    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        python_model=python_model,
    )

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_path)

    stream_result = loaded_pyfunc_model.predict_stream("single-input")
    assert isinstance(stream_result, types.GeneratorType)

    assert list(stream_result) == ["test1", "test2"]


def test_streamable_model_save_load(tmp_path):
    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")

    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        python_model="tests/pyfunc/sample_code/streamable_model_code.py",
    )

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_path)

    stream_result = loaded_pyfunc_model.predict_stream("single-input")
    assert isinstance(stream_result, types.GeneratorType)

    assert list(stream_result) == ["test1", "test2"]


def test_model_save_load_with_resources(tmp_path):
    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")
    pyfunc_model_path_2 = os.path.join(tmp_path, "pyfunc_model_2")

    expected_resources = {
        "api_version": "1",
        "databricks": {
            "serving_endpoint": [
                {"name": "databricks-mixtral-8x7b-instruct"},
                {"name": "databricks-bge-large-en"},
                {"name": "azure-eastus-model-serving-2_vs_endpoint"},
            ],
            "vector_search_index": [{"name": "rag.studio_bugbash.databricks_docs_index"}],
            "sql_warehouse": [{"name": "testid"}],
            "function": [
                {"name": "rag.studio.test_function_a"},
                {"name": "rag.studio.test_function_b"},
            ],
            "genie_space": [{"name": "genie_space_id_1"}, {"name": "genie_space_id_2"}],
            "uc_connection": [{"name": "test_connection_1"}, {"name": "test_connection_2"}],
            "table": [{"name": "rag.studio.table_a"}, {"name": "rag.studio.table_b"}],
            "app": [{"name": "test_databricks_app"}],
            "lakebase": [{"name": "test_databricks_lakebase"}],
        },
    }
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        conda_env=_conda_env(),
        python_model=mlflow.pyfunc.model.PythonModel(),
        resources=[
            DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
            DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
            DatabricksServingEndpoint(endpoint_name="azure-eastus-model-serving-2_vs_endpoint"),
            DatabricksVectorSearchIndex(index_name="rag.studio_bugbash.databricks_docs_index"),
            DatabricksSQLWarehouse(warehouse_id="testid"),
            DatabricksFunction(function_name="rag.studio.test_function_a"),
            DatabricksFunction(function_name="rag.studio.test_function_b"),
            DatabricksGenieSpace(genie_space_id="genie_space_id_1"),
            DatabricksGenieSpace(genie_space_id="genie_space_id_2"),
            DatabricksUCConnection(connection_name="test_connection_1"),
            DatabricksUCConnection(connection_name="test_connection_2"),
            DatabricksTable(table_name="rag.studio.table_a"),
            DatabricksTable(table_name="rag.studio.table_b"),
            DatabricksApp(app_name="test_databricks_app"),
            DatabricksLakebase(database_instance_name="test_databricks_lakebase"),
        ],
    )

    reloaded_model = Model.load(pyfunc_model_path)
    assert reloaded_model.resources == expected_resources

    yaml_file = tmp_path.joinpath("resources.yaml")
    with open(yaml_file, "w") as f:
        f.write(
            """
            api_version: "1"
            databricks:
                vector_search_index:
                - name: rag.studio_bugbash.databricks_docs_index
                serving_endpoint:
                - name: databricks-mixtral-8x7b-instruct
                - name: databricks-bge-large-en
                - name: azure-eastus-model-serving-2_vs_endpoint
                sql_warehouse:
                - name: testid
                function:
                - name: rag.studio.test_function_a
                - name: rag.studio.test_function_b
                lakebase:
                - name: test_databricks_lakebase
                genie_space:
                - name: genie_space_id_1
                - name: genie_space_id_2
                uc_connection:
                - name: test_connection_1
                - name: test_connection_2
                table:
                - name: rag.studio.table_a
                - name: rag.studio.table_b
                app:
                - name: test_databricks_app
            """
        )

    mlflow.pyfunc.save_model(
        path=pyfunc_model_path_2,
        conda_env=_conda_env(),
        python_model=mlflow.pyfunc.model.PythonModel(),
        resources=yaml_file,
    )
    reloaded_model = Model.load(pyfunc_model_path_2)
    assert reloaded_model.resources == expected_resources


def test_model_save_load_with_invokers_resources(tmp_path):
    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")
    pyfunc_model_path_2 = os.path.join(tmp_path, "pyfunc_model_2")

    expected_resources = {
        "api_version": "1",
        "databricks": {
            "serving_endpoint": [
                {"name": "databricks-mixtral-8x7b-instruct", "on_behalf_of_user": True},
                {"name": "databricks-bge-large-en"},
                {"name": "azure-eastus-model-serving-2_vs_endpoint"},
            ],
            "vector_search_index": [
                {"name": "rag.studio_bugbash.databricks_docs_index", "on_behalf_of_user": True}
            ],
            "sql_warehouse": [{"name": "testid"}],
            "function": [
                {"name": "rag.studio.test_function_a", "on_behalf_of_user": True},
                {"name": "rag.studio.test_function_b"},
            ],
            "genie_space": [
                {"name": "genie_space_id_1", "on_behalf_of_user": True},
                {"name": "genie_space_id_2"},
            ],
            "uc_connection": [{"name": "test_connection_1"}, {"name": "test_connection_2"}],
            "table": [
                {"name": "rag.studio.table_a", "on_behalf_of_user": True},
                {"name": "rag.studio.table_b"},
            ],
            "app": [{"name": "test_databricks_app"}],
            "lakebase": [{"name": "test_databricks_lakebase"}],
        },
    }
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        conda_env=_conda_env(),
        python_model=mlflow.pyfunc.model.PythonModel(),
        resources=[
            DatabricksServingEndpoint(
                endpoint_name="databricks-mixtral-8x7b-instruct", on_behalf_of_user=True
            ),
            DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
            DatabricksServingEndpoint(endpoint_name="azure-eastus-model-serving-2_vs_endpoint"),
            DatabricksVectorSearchIndex(
                index_name="rag.studio_bugbash.databricks_docs_index", on_behalf_of_user=True
            ),
            DatabricksSQLWarehouse(warehouse_id="testid"),
            DatabricksFunction(function_name="rag.studio.test_function_a", on_behalf_of_user=True),
            DatabricksFunction(function_name="rag.studio.test_function_b"),
            DatabricksGenieSpace(genie_space_id="genie_space_id_1", on_behalf_of_user=True),
            DatabricksGenieSpace(genie_space_id="genie_space_id_2"),
            DatabricksUCConnection(connection_name="test_connection_1"),
            DatabricksUCConnection(connection_name="test_connection_2"),
            DatabricksTable(table_name="rag.studio.table_a", on_behalf_of_user=True),
            DatabricksTable(table_name="rag.studio.table_b"),
            DatabricksApp(app_name="test_databricks_app"),
            DatabricksLakebase(database_instance_name="test_databricks_lakebase"),
        ],
    )

    reloaded_model = Model.load(pyfunc_model_path)
    assert reloaded_model.resources == expected_resources

    yaml_file = tmp_path.joinpath("resources.yaml")
    with open(yaml_file, "w") as f:
        f.write(
            """
            api_version: "1"
            databricks:
                vector_search_index:
                - name: rag.studio_bugbash.databricks_docs_index
                  on_behalf_of_user: True
                serving_endpoint:
                - name: databricks-mixtral-8x7b-instruct
                  on_behalf_of_user: True
                - name: databricks-bge-large-en
                - name: azure-eastus-model-serving-2_vs_endpoint
                sql_warehouse:
                - name: testid
                function:
                - name: rag.studio.test_function_a
                  on_behalf_of_user: True
                - name: rag.studio.test_function_b
                lakebase:
                - name: test_databricks_lakebase
                genie_space:
                - name: genie_space_id_1
                  on_behalf_of_user: True
                - name: genie_space_id_2
                uc_connection:
                - name: test_connection_1
                - name: test_connection_2
                table:
                - name: rag.studio.table_a
                  on_behalf_of_user: True
                - name: rag.studio.table_b
                app:
                - name: test_databricks_app
            """
        )

    mlflow.pyfunc.save_model(
        path=pyfunc_model_path_2,
        conda_env=_conda_env(),
        python_model=mlflow.pyfunc.model.PythonModel(),
        resources=yaml_file,
    )

    reloaded_model = Model.load(pyfunc_model_path_2)
    assert reloaded_model.resources == expected_resources


def test_model_log_with_invokers_resources(tmp_path):
    pyfunc_artifact_path = "pyfunc_model"

    expected_resources = {
        "api_version": "1",
        "databricks": {
            "serving_endpoint": [
                {"name": "databricks-mixtral-8x7b-instruct"},
                {"name": "databricks-bge-large-en", "on_behalf_of_user": True},
                {"name": "azure-eastus-model-serving-2_vs_endpoint"},
            ],
            "vector_search_index": [
                {"name": "rag.studio_bugbash.databricks_docs_index", "on_behalf_of_user": True}
            ],
            "sql_warehouse": [{"name": "testid", "on_behalf_of_user": True}],
            "function": [
                {"name": "rag.studio.test_function_a"},
                {"name": "rag.studio.test_function_b", "on_behalf_of_user": True},
            ],
            "genie_space": [
                {"name": "genie_space_id_1"},
                {"name": "genie_space_id_2", "on_behalf_of_user": True},
            ],
            "uc_connection": [
                {"name": "test_connection_1"},
                {"name": "test_connection_2", "on_behalf_of_user": True},
            ],
            "table": [
                {"name": "rag.studio.table_a"},
                {"name": "rag.studio.table_b", "on_behalf_of_user": True},
            ],
        },
    }
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name=pyfunc_artifact_path,
            python_model=mlflow.pyfunc.model.PythonModel(),
            resources=[
                DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
                DatabricksServingEndpoint(
                    endpoint_name="databricks-bge-large-en", on_behalf_of_user=True
                ),
                DatabricksServingEndpoint(endpoint_name="azure-eastus-model-serving-2_vs_endpoint"),
                DatabricksVectorSearchIndex(
                    index_name="rag.studio_bugbash.databricks_docs_index", on_behalf_of_user=True
                ),
                DatabricksSQLWarehouse(warehouse_id="testid", on_behalf_of_user=True),
                DatabricksFunction(function_name="rag.studio.test_function_a"),
                DatabricksFunction(
                    function_name="rag.studio.test_function_b", on_behalf_of_user=True
                ),
                DatabricksGenieSpace(genie_space_id="genie_space_id_1"),
                DatabricksGenieSpace(genie_space_id="genie_space_id_2", on_behalf_of_user=True),
                DatabricksUCConnection(connection_name="test_connection_1"),
                DatabricksUCConnection(connection_name="test_connection_2", on_behalf_of_user=True),
                DatabricksTable(table_name="rag.studio.table_a"),
                DatabricksTable(table_name="rag.studio.table_b", on_behalf_of_user=True),
            ],
        )
    pyfunc_model_uri = f"runs:/{run.info.run_id}/{pyfunc_artifact_path}"
    pyfunc_model_path = _download_artifact_from_uri(pyfunc_model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert reloaded_model.resources == expected_resources

    yaml_file = tmp_path.joinpath("resources.yaml")
    with open(yaml_file, "w") as f:
        f.write(
            """
            api_version: "1"
            databricks:
                vector_search_index:
                - name: rag.studio_bugbash.databricks_docs_index
                  on_behalf_of_user: True
                serving_endpoint:
                - name: databricks-mixtral-8x7b-instruct
                - name: databricks-bge-large-en
                  on_behalf_of_user: True
                - name: azure-eastus-model-serving-2_vs_endpoint
                sql_warehouse:
                - name: testid
                  on_behalf_of_user: True
                function:
                - name: rag.studio.test_function_a
                - name: rag.studio.test_function_b
                  on_behalf_of_user: True
                genie_space:
                - name: genie_space_id_1
                - name: genie_space_id_2
                  on_behalf_of_user: True
                uc_connection:
                - name: test_connection_1
                - name: test_connection_2
                  on_behalf_of_user: True
                table:
                - name: "rag.studio.table_a"
                - name: "rag.studio.table_b"
                  on_behalf_of_user: True
            """
        )

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name=pyfunc_artifact_path,
            python_model=mlflow.pyfunc.model.PythonModel(),
            resources=yaml_file,
        )
    pyfunc_model_uri = f"runs:/{run.info.run_id}/{pyfunc_artifact_path}"
    pyfunc_model_path = _download_artifact_from_uri(pyfunc_model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert reloaded_model.resources == expected_resources


def test_model_log_with_resources(tmp_path):
    pyfunc_artifact_path = "pyfunc_model"

    expected_resources = {
        "api_version": "1",
        "databricks": {
            "serving_endpoint": [
                {"name": "databricks-mixtral-8x7b-instruct"},
                {"name": "databricks-bge-large-en"},
                {"name": "azure-eastus-model-serving-2_vs_endpoint"},
            ],
            "vector_search_index": [{"name": "rag.studio_bugbash.databricks_docs_index"}],
            "sql_warehouse": [{"name": "testid"}],
            "function": [
                {"name": "rag.studio.test_function_a"},
                {"name": "rag.studio.test_function_b"},
            ],
            "genie_space": [
                {"name": "genie_space_id_1"},
                {"name": "genie_space_id_2"},
            ],
            "uc_connection": [{"name": "test_connection_1"}, {"name": "test_connection_2"}],
            "table": [{"name": "rag.studio.table_a"}, {"name": "rag.studio.table_b"}],
            "app": [{"name": "test_databricks_app"}],
            "lakebase": [{"name": "test_databricks_lakebase"}],
        },
    }
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name=pyfunc_artifact_path,
            python_model=mlflow.pyfunc.model.PythonModel(),
            resources=[
                DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
                DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
                DatabricksServingEndpoint(endpoint_name="azure-eastus-model-serving-2_vs_endpoint"),
                DatabricksVectorSearchIndex(index_name="rag.studio_bugbash.databricks_docs_index"),
                DatabricksSQLWarehouse(warehouse_id="testid"),
                DatabricksFunction(function_name="rag.studio.test_function_a"),
                DatabricksFunction(function_name="rag.studio.test_function_b"),
                DatabricksGenieSpace(genie_space_id="genie_space_id_1"),
                DatabricksGenieSpace(genie_space_id="genie_space_id_2"),
                DatabricksUCConnection(connection_name="test_connection_1"),
                DatabricksUCConnection(connection_name="test_connection_2"),
                DatabricksTable(table_name="rag.studio.table_a"),
                DatabricksTable(table_name="rag.studio.table_b"),
                DatabricksApp(app_name="test_databricks_app"),
                DatabricksLakebase(database_instance_name="test_databricks_lakebase"),
            ],
        )
    pyfunc_model_uri = f"runs:/{run.info.run_id}/{pyfunc_artifact_path}"
    pyfunc_model_path = _download_artifact_from_uri(pyfunc_model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert reloaded_model.resources == expected_resources

    yaml_file = tmp_path.joinpath("resources.yaml")
    with open(yaml_file, "w") as f:
        f.write(
            """
            api_version: "1"
            databricks:
                vector_search_index:
                - name: rag.studio_bugbash.databricks_docs_index
                serving_endpoint:
                - name: databricks-mixtral-8x7b-instruct
                - name: databricks-bge-large-en
                - name: azure-eastus-model-serving-2_vs_endpoint
                sql_warehouse:
                - name: testid
                function:
                - name: rag.studio.test_function_a
                - name: rag.studio.test_function_b
                lakebase:
                - name: test_databricks_lakebase
                genie_space:
                - name: genie_space_id_1
                - name: genie_space_id_2
                uc_connection:
                - name: test_connection_1
                - name: test_connection_2
                table:
                - name: "rag.studio.table_a"
                - name: "rag.studio.table_b"
                app:
                - name: test_databricks_app
            """
        )

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name=pyfunc_artifact_path,
            python_model=mlflow.pyfunc.model.PythonModel(),
            resources=yaml_file,
        )
    pyfunc_model_uri = f"runs:/{run.info.run_id}/{pyfunc_artifact_path}"
    pyfunc_model_path = _download_artifact_from_uri(pyfunc_model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert reloaded_model.resources == expected_resources


def test_pyfunc_as_code_log_and_load():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model="tests/pyfunc/sample_code/python_model.py",
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    model_input = "asdf"
    expected_output = f"This was the input: {model_input}"
    assert loaded_model.predict(model_input) == expected_output


def test_pyfunc_as_code_log_and_load_with_path():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=Path("tests/pyfunc/sample_code/python_model.py"),
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    model_input = "asdf"
    expected_output = f"This was the input: {model_input}"
    assert loaded_model.predict(model_input) == expected_output


def test_pyfunc_as_code_with_config(tmp_path):
    temp_file = tmp_path / "config.yml"
    temp_file.write_text("timeout: 400")

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model="tests/pyfunc/sample_code/python_model_with_config.py",
            model_config=str(temp_file),
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    model_input = "input"
    expected_output = f"Predict called with input {model_input}, timeout 400"
    assert loaded_model.predict(model_input) == expected_output


def test_pyfunc_as_code_with_path_config(tmp_path):
    temp_file = tmp_path / "config.yml"
    temp_file.write_text("timeout: 400")

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model="tests/pyfunc/sample_code/python_model_with_config.py",
            model_config=temp_file,
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    model_input = "input"
    expected_output = f"Predict called with input {model_input}, timeout 400"
    assert loaded_model.predict(model_input) == expected_output


def test_pyfunc_as_code_with_dict_config():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model="tests/pyfunc/sample_code/python_model_with_config.py",
            model_config={"timeout": 400},
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    model_input = "input"
    expected_output = f"Predict called with input {model_input}, timeout 400"
    assert loaded_model.predict(model_input) == expected_output


def test_pyfunc_as_code_log_and_load_with_code_paths():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model="tests/pyfunc/sample_code/python_model_with_utils.py",
            code_paths=["tests/pyfunc/sample_code/utils.py"],
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    model_input = "asdf"
    expected_output = f"My utils function received this input: {model_input}"
    assert loaded_model.predict(model_input) == expected_output


def test_pyfunc_as_code_with_dependencies():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model="tests/pyfunc/sample_code/code_with_dependencies.py",
            pip_requirements=["pandas"],
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    model_input = "user_123"
    expected_output = f"Input: {model_input}. Retriever called with ID: {model_input}. Output: 42."
    assert loaded_model.predict(model_input) == expected_output

    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert reloaded_model.metadata["dependencies_schemas"] == {
        "retrievers": [
            {
                "doc_uri": "doc-uri",
                "name": "retriever",
                "other_columns": ["column1", "column2"],
                "primary_key": "primary-key",
                "text_column": "text-column",
            }
        ]
    }


@pytest.mark.parametrize("is_in_db_model_serving", ["true", "false"])
@pytest.mark.parametrize("stream", [True, False])
def test_pyfunc_as_code_with_dependencies_store_dependencies_schemas_in_trace(
    monkeypatch, is_in_db_model_serving, stream
):
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", is_in_db_model_serving)
    monkeypatch.setenv("ENABLE_MLFLOW_TRACING", "true")
    is_in_db_model_serving = is_in_db_model_serving == "true"
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model="tests/pyfunc/sample_code/code_with_dependencies.py",
            pip_requirements=["pandas"],
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    model_input = "user_123"
    expected_output = f"Input: {model_input}. Retriever called with ID: {model_input}. Output: 42."
    func = loaded_model.predict_stream if stream else loaded_model.predict

    def _get_result(output):
        return list(output)[0] if stream else output

    if is_in_db_model_serving:
        with set_prediction_context(Context(request_id="1234")):
            assert _get_result(func(model_input)) == expected_output
    else:
        assert _get_result(func(model_input)) == expected_output

    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    expected_dependencies_schemas = {
        DependenciesSchemasType.RETRIEVERS.value: [
            {
                "doc_uri": "doc-uri",
                "name": "retriever",
                "other_columns": ["column1", "column2"],
                "primary_key": "primary-key",
                "text_column": "text-column",
            }
        ]
    }
    assert reloaded_model.metadata["dependencies_schemas"] == expected_dependencies_schemas

    if is_in_db_model_serving:
        trace_dict = pop_trace("1234")
        trace = Trace.from_dict(trace_dict)
        assert trace.info.trace_id.startswith("tr-")
        assert trace.info.client_request_id == "1234"
    else:
        trace = get_traces()[0]
    assert trace.info.tags[DependenciesSchemasType.RETRIEVERS.value] == json.dumps(
        expected_dependencies_schemas[DependenciesSchemasType.RETRIEVERS.value]
    )


@pytest.mark.parametrize("stream", [True, False])
def test_no_traces_collected_for_pyfunc_as_code_with_dependencies_if_no_tracing_enabled(
    monkeypatch, stream
):
    # This sets model without trace inside code_with_dependencies.py file
    monkeypatch.setenv("TEST_TRACE", "false")
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model="tests/pyfunc/sample_code/code_with_dependencies.py",
            pip_requirements=["pandas"],
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    model_input = "user_123"
    expected_output = f"Input: {model_input}. Retriever called with ID: {model_input}. Output: 42."
    if stream:
        assert next(loaded_model.predict_stream(model_input)) == expected_output
    else:
        assert loaded_model.predict(model_input) == expected_output

    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri)
    reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    expected_dependencies_schemas = {
        DependenciesSchemasType.RETRIEVERS.value: [
            {
                "doc_uri": "doc-uri",
                "name": "retriever",
                "other_columns": ["column1", "column2"],
                "primary_key": "primary-key",
                "text_column": "text-column",
            }
        ]
    }
    assert reloaded_model.metadata["dependencies_schemas"] == expected_dependencies_schemas

    # no traces will be logged at all
    traces = get_traces()
    assert len(traces) == 0


def test_pyfunc_as_code_log_and_load_wrong_path():
    with pytest.raises(
        MlflowException,
        match="The provided model path",
    ):
        with mlflow.start_run():
            mlflow.pyfunc.log_model(
                name="model",
                python_model="asdf",
            )


def test_predict_as_code():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model="tests/pyfunc/sample_code/func_code.py",
            input_example=["string"],
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    model_input = "asdf"
    expected_output = pd.DataFrame([model_input])
    pandas.testing.assert_frame_equal(loaded_model.predict([model_input]), expected_output)


def test_predict_as_code_with_type_hint():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model="tests/pyfunc/sample_code/func_code_with_type_hint.py",
            input_example=["string"],
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    model_input = "asdf"
    expected_output = [model_input]
    assert loaded_model.predict([model_input]) == expected_output


def test_predict_as_code_with_config():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model="tests/pyfunc/sample_code/func_code_with_config.py",
            input_example=["string"],
            model_config="tests/pyfunc/sample_code/config.yml",
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    model_input = "asdf"
    expected_output = f"This was the input: {model_input}, timeout 300"
    assert loaded_model.predict([model_input]) == expected_output


def test_model_as_code_pycache_cleaned_up():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model="tests/pyfunc/sample_code/python_model.py",
        )

    path = _download_artifact_from_uri(model_info.model_uri)
    assert list(Path(path).rglob("__pycache__")) == []


def test_model_pip_requirements_pin_numpy_when_pandas_included():
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            import pandas as pd  # noqa: F401

            return model_input

    expected_mlflow_version = _mlflow_major_version_string()

    # no numpy when pandas > 2.1.2
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=TestModel(), input_example="abc"
        )

        _assert_pip_requirements(
            model_info.model_uri,
            [
                expected_mlflow_version,
                f"cloudpickle=={importlib.metadata.version('cloudpickle')}",
                f"pandas=={importlib.metadata.version('pandas')}",
            ],
            strict=True,
        )

    original_get_installed_version = _get_installed_version

    def mock_get_installed_version(package, module=None):
        if package == "pandas":
            return "2.1.0"
        return original_get_installed_version(package, module)

    # include numpy when pandas < 2.1.2
    with (
        mlflow.start_run(),
        mock.patch(
            "mlflow.utils.requirements_utils._get_installed_version",
            side_effect=mock_get_installed_version,
        ),
    ):
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=TestModel(), input_example="abc"
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [
                expected_mlflow_version,
                "pandas==2.1.0",
                f"numpy=={np.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
            ],
            strict=True,
        )

    # no input_example, so pandas not included in requirements
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(name="model", python_model=TestModel())
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, f"cloudpickle=={cloudpickle.__version__}"],
            strict=True,
        )


def test_environment_variables_used_during_model_logging(monkeypatch):
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            monkeypatch.setenv("TEST_API_KEY", "test_env")
            monkeypatch.setenv("ANOTHER_API_KEY", "test_env")
            monkeypatch.setenv("INVALID_ENV_VAR", "var")
            # existing env var is tracked
            os.environ["TEST_API_KEY"]
            # existing env var fetched by getenv is tracked
            os.getenv("ANOTHER_API_KEY")
            # existing env var not in allowlist is not tracked
            os.environ.get("INVALID_ENV_VAR")
            # non-existing env var is not tracked
            os.environ.get("INVALID_API_KEY")
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=MyModel(), input_example="data"
        )
    assert "TEST_API_KEY" in model_info.env_vars
    assert "ANOTHER_API_KEY" in model_info.env_vars
    assert "INVALID_ENV_VAR" not in model_info.env_vars
    assert "INVALID_API_KEY" not in model_info.env_vars
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_model.metadata.env_vars == model_info.env_vars

    # if no input_example provided, we do not run predict, and no env vars are captured
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(name="model", python_model=MyModel())
    assert model_info.env_vars is None

    # disable logging by setting environment variable
    monkeypatch.setenv(MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING.name, "false")
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=MyModel(), input_example="data"
        )
    assert model_info.env_vars is None


def test_pyfunc_model_without_context_in_predict():
    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, model_input, params=None):
            return model_input

        def predict_stream(self, model_input, params=None):
            yield model_input

    m = Model()
    assert m.predict("abc") == "abc"
    assert next(iter(m.predict_stream("abc"))) == "abc"

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(name="model", python_model=m, input_example="abc")
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_model.predict("abc") is not None
    assert next(iter(pyfunc_model.predict_stream("abc"))) is not None


def test_callable_python_model_without_context_in_predict():
    def predict(model_input):
        return model_input

    assert predict("abc") == "abc"
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=predict, input_example="abc"
        )
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_model.predict("abc") is not None


def test_pyfunc_model_with_wrong_predict_signature_warning():
    with pytest.warns(
        FutureWarning,
        match=r"Model's `predict` method contains invalid parameters: {'messages'}",
    ):

        class Model(mlflow.pyfunc.PythonModel):
            def predict(self, context, messages, params=None):
                return messages

    with pytest.warns(
        FutureWarning,
        match=r"Model's `predict_stream` method contains invalid parameters: {'_'}",
    ):

        class Model(mlflow.pyfunc.PythonModel):
            def predict(self, model_input, params=None):
                return model_input

            def predict_stream(self, _, model_input, params=None):
                yield model_input


def test_pyfunc_model_input_example_with_signature():
    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    signature = infer_signature(["a", "b", "c"])
    with mlflow.start_run():
        with pytest.warns(
            UserWarning, match=r"An input example was not provided when logging the model"
        ):
            mlflow.pyfunc.log_model(name="model", python_model=Model(), signature=signature)

    with mlflow.start_run():
        with pytest.raises(
            MlflowException, match=r"Input example does not match the model signature"
        ):
            mlflow.pyfunc.log_model(
                name="model", python_model=Model(), signature=signature, input_example=123
            )


@pytest.mark.parametrize("save_model", [True, False])
@pytest.mark.parametrize("use_user_auth_policy", [True, False])
@pytest.mark.parametrize("use_system_policy", [True, False])
def test_model_log_with_auth_policy(tmp_path, save_model, use_user_auth_policy, use_system_policy):
    pyfunc_save_artifact_path = os.path.join(tmp_path, "pyfunc_model_save")
    pyfunc_log_artifact_path = "pyfunc_model_log"

    expected_auth_policy = {"system_auth_policy": {}, "user_auth_policy": {}}

    system_auth_policy = None
    if use_system_policy:
        system_auth_policy = SystemAuthPolicy(
            resources=[
                DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
                DatabricksVectorSearchIndex(index_name="rag.studio_bugbash.databricks_docs_index"),
                DatabricksFunction(function_name="rag.studio.test_function_a"),
                DatabricksUCConnection(connection_name="test_connection_1"),
            ]
        )
        expected_auth_policy["system_auth_policy"] = {
            "resources": {
                "api_version": "1",
                "databricks": {
                    "function": [{"name": "rag.studio.test_function_a"}],
                    "serving_endpoint": [{"name": "databricks-mixtral-8x7b-instruct"}],
                    "uc_connection": [{"name": "test_connection_1"}],
                    "vector_search_index": [{"name": "rag.studio_bugbash.databricks_docs_index"}],
                },
            }
        }

    user_auth_policy = None
    if use_user_auth_policy:
        user_auth_policy = UserAuthPolicy(
            api_scopes=[
                "catalog.catalogs",
                "vectorsearch.vector-search-indexes",
                "workspace.workspace",
            ]
        )
        expected_auth_policy["user_auth_policy"] = {
            "api_scopes": [
                "catalog.catalogs",
                "vectorsearch.vector-search-indexes",
                "workspace.workspace",
            ]
        }

    auth_policy = AuthPolicy(
        user_auth_policy=user_auth_policy, system_auth_policy=system_auth_policy
    )

    if save_model:
        mlflow.pyfunc.save_model(
            path=pyfunc_save_artifact_path,
            conda_env=_conda_env(),
            python_model=mlflow.pyfunc.model.PythonModel(),
            auth_policy=auth_policy,
        )
        reloaded_model = Model.load(pyfunc_save_artifact_path)
    else:
        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(
                name=pyfunc_log_artifact_path,
                python_model=mlflow.pyfunc.model.PythonModel(),
                auth_policy=auth_policy,
            )

        pyfunc_model_uri = f"runs:/{run.info.run_id}/{pyfunc_log_artifact_path}"
        pyfunc_model_path = _download_artifact_from_uri(pyfunc_model_uri)
        reloaded_model = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))

    assert reloaded_model.auth_policy == expected_auth_policy


def test_both_resources_and_auth_policy():
    pyfunc_log_artifact_path = "pyfunc_model_log"
    system_auth_policy = SystemAuthPolicy(
        resources=[DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct")]
    )
    user_auth_policy = UserAuthPolicy(api_scopes=["workspace.workspace"])
    auth_policy = AuthPolicy(
        user_auth_policy=user_auth_policy, system_auth_policy=system_auth_policy
    )

    with mlflow.start_run() as _:
        with pytest.raises(
            ValueError, match="Only one of `resources`, and `auth_policy` can be specified."
        ):
            mlflow.pyfunc.log_model(
                name=pyfunc_log_artifact_path,
                python_model=mlflow.pyfunc.model.PythonModel(),
                auth_policy=auth_policy,
                resources=[
                    DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct")
                ],
            )


@pytest.mark.parametrize("compression", ["lzma", "bzip2", "gzip"])
def test_model_save_load_compression(
    monkeypatch, sklearn_knn_model, main_scoped_model_class, iris_data, tmp_path, compression
):
    monkeypatch.setenv(MLFLOW_LOG_MODEL_COMPRESSION.name, compression)
    sklearn_model_path = os.path.join(tmp_path, "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(tmp_path, "pyfunc_model")

    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        conda_env=_conda_env(),
        python_model=main_scoped_model_class(test_predict),
    )

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_path)
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict(iris_data[0]),
        test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]),
    )


@pytest.mark.skip(reason="Enable once we re-enable the warning")
def test_load_model_warning():
    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: list[str]):
            return model_input

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            python_model=Model(),
            name="model",
            input_example=["a", "b", "c"],
        )

    with pytest.warns(UserWarning, match=r"`runs:/<run_id>/artifact_path` is deprecated"):
        mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")


def test_pyfunc_model_traces_link_to_model_id():
    class TestModel(mlflow.pyfunc.PythonModel):
        @mlflow.trace
        def predict(self, model_input: list[str]) -> list[str]:
            return model_input

    model_infos = []
    for i in range(3):
        model_infos.append(
            mlflow.pyfunc.log_model(
                name="test_model",
                python_model=TestModel(),
                input_example=["a", "b", "c"],
            )
        )

    for model_info in model_infos:
        pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
        pyfunc_model.predict(["a", "b", "c"])

    traces = get_traces()[::-1]
    assert len(traces) == 3
    for i in range(3):
        assert traces[i].info.request_metadata[TraceMetadataKey.MODEL_ID] == model_infos[i].model_id


class ExampleModel(mlflow.pyfunc.PythonModel):
    def predict(self, model_input: list[str]) -> list[str]:
        return model_input


def test_lock_model_requirements(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("MLFLOW_LOCK_MODEL_DEPENDENCIES", "true")

    model_info = mlflow.pyfunc.log_model(name="model", python_model=ExampleModel())
    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri, output_path=tmp_path)
    requirements_txt = next(Path(pyfunc_model_path).rglob("requirements.txt"))
    requirements_txt_contents = requirements_txt.read_text()
    assert "# Locked requirements" in requirements_txt_contents
    assert "mlflow==" in requirements_txt_contents
    assert "packaging==" in requirements_txt_contents
    # Check that pip can install the locked requirements
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--ignore-installed",
            "--dry-run",
            "--requirement",
            requirements_txt,
        ],
    )
    # Check that conda environment can be created with the locked requirements
    conda_yaml = next(Path(pyfunc_model_path).rglob("conda.yaml"))
    conda_yaml_contents = conda_yaml.read_text()
    assert "# Locked requirements" in conda_yaml_contents
    assert "mlflow==" in requirements_txt_contents
    assert "packaging==" in conda_yaml_contents
    subprocess.check_call(
        [
            "conda",
            "env",
            "create",
            "--file",
            conda_yaml,
            "--dry-run",
            "--yes",
        ],
    )


def test_lock_model_requirements_pip_requirements(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("MLFLOW_LOCK_MODEL_DEPENDENCIES", "true")
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=ExampleModel(),
        pip_requirements=["openai"],
    )
    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri, output_path=tmp_path)
    requirements_txt = next(Path(pyfunc_model_path).rglob("requirements.txt"))
    contents = requirements_txt.read_text()
    assert "# Locked requirements" in contents
    assert "mlflow==" in contents
    assert "openai==" in contents
    assert "httpx==" in contents


def test_lock_model_requirements_extra_pip_requirements(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("MLFLOW_LOCK_MODEL_DEPENDENCIES", "true")
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=ExampleModel(),
        extra_pip_requirements=["openai"],
    )
    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri, output_path=tmp_path)
    requirements_txt = next(Path(pyfunc_model_path).rglob("requirements.txt"))
    contents = requirements_txt.read_text()
    assert "# Locked requirements" in contents
    assert "mlflow==" in contents
    assert "openai==" in contents
    assert "httpx==" in contents


def test_lock_model_requirements_constraints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    constraints_file = tmp_path / "constraints.txt"
    constraints_file.write_text("openai==1.82.0")
    monkeypatch.setenv("MLFLOW_LOCK_MODEL_DEPENDENCIES", "true")
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=ExampleModel(),
        pip_requirements=["openai", f"-c {constraints_file}"],
    )
    pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri, output_path=tmp_path)
    requirements_txt = next(Path(pyfunc_model_path).rglob("requirements.txt"))
    contents = requirements_txt.read_text()
    assert "# Locked requirements" in contents
    assert "mlflow==" in contents
    assert "openai==1.82.0" in contents
    assert "httpx==" in contents


@pytest.mark.parametrize(
    ("input_example", "expected_result"), [(["Hello", "World"], True), (None, False)]
)
def test_load_context_with_input_example(input_example, expected_result):
    class MyModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            raise Exception("load_context was called")

        def predict(self, model_input: list[str], params=None):
            return model_input

    msg = "Failed to run the predict function on input example"

    with mock.patch("mlflow.models.signature._logger.warning") as mock_warning:
        mlflow.pyfunc.log_model(
            name="model",
            python_model=MyModel(),
            input_example=input_example,
        )
        assert any(msg in call.args[0] for call in mock_warning.call_args_list) == expected_result
