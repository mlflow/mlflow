import cloudpickle
import os
import json
from subprocess import Popen, PIPE
from unittest import mock

import numpy as np
import pandas as pd
import pandas.testing
import pytest
import sklearn.datasets
import sklearn.linear_model
import sklearn.neighbors
import yaml

import mlflow
import mlflow.pyfunc
import mlflow.pyfunc.model
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
from mlflow.models.utils import _read_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import (
    get_artifact_uri as utils_get_artifact_uri,
    _download_artifact_from_uri,
)
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

import tests
from tests.helper_functions import pyfunc_serve_and_score_model
from tests.helper_functions import (
    _compare_conda_env_requirements,
    _assert_pip_requirements,
)
from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import


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
            # pylint: disable=attribute-defined-outside-init
            self.model = mlflow.sklearn.load_model(model_uri=context.artifacts["sk_model"])

        def predict(self, context, model_input):
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
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture
def pyfunc_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(
        conda_env,
        additional_pip_deps=["scikit-learn", "pytest", "cloudpickle"],
    )
    return conda_env


def _conda_env():
    # NB: We need mlflow as a dependency in the environment.
    return _mlflow_conda_env(
        additional_pip_deps=[
            "cloudpickle=={}".format(cloudpickle.__version__),
            "scikit-learn=={}".format(sklearn.__version__),
        ],
    )


@pytest.mark.large
def test_model_save_load(sklearn_knn_model, main_scoped_model_class, iris_data, tmpdir):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")

    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        conda_env=_conda_env(),
        python_model=main_scoped_model_class(test_predict),
    )

    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(model_uri=pyfunc_model_path)
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict(iris_data[0]),
        test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]),
    )


@pytest.mark.large
def test_pyfunc_model_log_load_no_active_run(sklearn_knn_model, main_scoped_model_class, iris_data):
    sklearn_artifact_path = "sk_model_no_run"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_knn_model, artifact_path=sklearn_artifact_path)
        sklearn_model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=sklearn_artifact_path
        )

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_artifact_path = "pyfunc_model"
    assert mlflow.active_run() is None
    mlflow.pyfunc.log_model(
        artifact_path=pyfunc_artifact_path,
        artifacts={"sk_model": sklearn_model_uri},
        python_model=main_scoped_model_class(test_predict),
    )
    pyfunc_model_uri = "runs:/{run_id}/{artifact_path}".format(
        run_id=mlflow.active_run().info.run_id, artifact_path=pyfunc_artifact_path
    )
    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(model_uri=pyfunc_model_uri)
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict(iris_data[0]),
        test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]),
    )
    mlflow.end_run()


@pytest.mark.large
def test_model_log_load(sklearn_knn_model, main_scoped_model_class, iris_data):
    sklearn_artifact_path = "sk_model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_knn_model, artifact_path=sklearn_artifact_path)
        sklearn_model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=sklearn_artifact_path
        )

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            artifacts={"sk_model": sklearn_model_uri},
            python_model=main_scoped_model_class(test_predict),
        )
        pyfunc_model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=pyfunc_artifact_path
        )
        pyfunc_model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=pyfunc_artifact_path
            )
        )
        model_config = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))

    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(model_uri=pyfunc_model_uri)
    assert model_config.to_yaml() == loaded_pyfunc_model.metadata.to_yaml()
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict(iris_data[0]),
        test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]),
    )


@pytest.mark.large
def test_signature_and_examples_are_saved_correctly(iris_data, main_scoped_model_class, tmpdir):
    sklearn_model_path = tmpdir.join("sklearn_model").strpath
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    data = iris_data
    signature_ = infer_signature(*data)
    example_ = data[0][
        :3,
    ]
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
                    assert np.array_equal(_read_example(mlflow_model, path), example)


def test_log_model_calls_register_model(sklearn_knn_model, main_scoped_model_class):
    register_model_patch = mock.patch("mlflow.register_model")
    with register_model_patch:
        sklearn_artifact_path = "sk_model_no_run"
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                sk_model=sklearn_knn_model, artifact_path=sklearn_artifact_path
            )
            sklearn_model_uri = "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=sklearn_artifact_path
            )

        def test_predict(sk_model, model_input):
            return sk_model.predict(model_input) * 2

        pyfunc_artifact_path = "pyfunc_model"
        assert mlflow.active_run() is None
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            artifacts={"sk_model": sklearn_model_uri},
            python_model=main_scoped_model_class(test_predict),
            registered_model_name="AdsModel1",
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=pyfunc_artifact_path
        )
        mlflow.register_model.assert_called_once_with(
            model_uri, "AdsModel1", await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )
        mlflow.end_run()


def test_log_model_no_registered_model_name(sklearn_knn_model, main_scoped_model_class):
    register_model_patch = mock.patch("mlflow.register_model")
    with register_model_patch:
        sklearn_artifact_path = "sk_model_no_run"
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                sk_model=sklearn_knn_model, artifact_path=sklearn_artifact_path
            )
            sklearn_model_uri = "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=sklearn_artifact_path
            )

        def test_predict(sk_model, model_input):
            return sk_model.predict(model_input) * 2

        pyfunc_artifact_path = "pyfunc_model"
        assert mlflow.active_run() is None
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            artifacts={"sk_model": sklearn_model_uri},
            python_model=main_scoped_model_class(test_predict),
        )
        mlflow.register_model.assert_not_called()
        mlflow.end_run()


@pytest.mark.large
def test_model_load_from_remote_uri_succeeds(
    sklearn_knn_model, main_scoped_model_class, tmpdir, mock_s3_bucket, iris_data
):
    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_repo = S3ArtifactRepository(artifact_root)

    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)
    sklearn_artifact_path = "sk_model"
    artifact_repo.log_artifacts(sklearn_model_path, artifact_path=sklearn_artifact_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(test_predict),
        conda_env=_conda_env(),
    )

    pyfunc_artifact_path = "pyfunc_model"
    artifact_repo.log_artifacts(pyfunc_model_path, artifact_path=pyfunc_artifact_path)

    model_uri = artifact_root + "/" + pyfunc_artifact_path
    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(model_uri=model_uri)
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict(iris_data[0]),
        test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]),
    )


@pytest.mark.large
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
    assert all([item in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME] for item in custom_kwargs])


@pytest.mark.large
def test_pyfunc_model_serving_without_conda_env_activation_succeeds_with_main_scoped_class(
    sklearn_knn_model, main_scoped_model_class, iris_data, tmpdir
):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(test_predict),
        conda_env=_conda_env(),
    )
    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(model_uri=pyfunc_model_path)

    sample_input = pd.DataFrame(iris_data[0])
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=pyfunc_model_path,
        data=sample_input,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        extra_args=["--no-conda"],
    )
    assert scoring_response.status_code == 200
    np.testing.assert_array_equal(
        np.array(json.loads(scoring_response.text)), loaded_pyfunc_model.predict(sample_input)
    )


@pytest.mark.large
def test_pyfunc_model_serving_with_conda_env_activation_succeeds_with_main_scoped_class(
    sklearn_knn_model, main_scoped_model_class, iris_data, tmpdir
):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(test_predict),
        conda_env=_conda_env(),
    )
    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(model_uri=pyfunc_model_path)

    sample_input = pd.DataFrame(iris_data[0])
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=pyfunc_model_path,
        data=sample_input,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    assert scoring_response.status_code == 200
    np.testing.assert_array_equal(
        np.array(json.loads(scoring_response.text)), loaded_pyfunc_model.predict(sample_input)
    )


@pytest.mark.large
def test_pyfunc_model_serving_without_conda_env_activation_succeeds_with_module_scoped_class(
    sklearn_knn_model, iris_data, tmpdir
):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=ModuleScopedSklearnModel(test_predict),
        code_path=[os.path.dirname(tests.__file__)],
        conda_env=_conda_env(),
    )
    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(model_uri=pyfunc_model_path)

    sample_input = pd.DataFrame(iris_data[0])
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=pyfunc_model_path,
        data=sample_input,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        extra_args=["--no-conda"],
    )
    assert scoring_response.status_code == 200
    np.testing.assert_array_equal(
        np.array(json.loads(scoring_response.text)), loaded_pyfunc_model.predict(sample_input)
    )


@pytest.mark.large
def test_pyfunc_cli_predict_command_without_conda_env_activation_succeeds(
    sklearn_knn_model, main_scoped_model_class, iris_data, tmpdir
):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(test_predict),
        conda_env=_conda_env(),
    )
    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(model_uri=pyfunc_model_path)

    sample_input = pd.DataFrame(iris_data[0])
    input_csv_path = os.path.join(str(tmpdir), "input with spaces.csv")
    sample_input.to_csv(input_csv_path, header=True, index=False)
    output_json_path = os.path.join(str(tmpdir), "output.json")
    process = Popen(
        [
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
            "--no-conda",
        ],
        stdout=PIPE,
        stderr=PIPE,
        preexec_fn=os.setsid,
    )
    _, stderr = process.communicate()
    assert 0 == process.wait(), "stderr = \n\n{}\n\n".format(stderr)

    result_df = pandas.read_json(output_json_path, orient="records")
    np.testing.assert_array_equal(
        result_df.values.transpose()[0], loaded_pyfunc_model.predict(sample_input)
    )


@pytest.mark.large
def test_pyfunc_cli_predict_command_with_conda_env_activation_succeeds(
    sklearn_knn_model, main_scoped_model_class, iris_data, tmpdir
):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(test_predict),
        conda_env=_conda_env(),
    )
    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(model_uri=pyfunc_model_path)

    sample_input = pd.DataFrame(iris_data[0])
    input_csv_path = os.path.join(str(tmpdir), "input with spaces.csv")
    sample_input.to_csv(input_csv_path, header=True, index=False)
    output_json_path = os.path.join(str(tmpdir), "output.json")
    process = Popen(
        [
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
    _, stderr = process.communicate()
    assert 0 == process.wait(), "stderr = \n\n{}\n\n".format(stderr)
    result_df = pandas.read_json(output_json_path, orient="records")
    np.testing.assert_array_equal(
        result_df.values.transpose()[0], loaded_pyfunc_model.predict(sample_input)
    )


@pytest.mark.large
def test_save_model_persists_specified_conda_env_in_mlflow_model_directory(
    sklearn_knn_model, main_scoped_model_class, pyfunc_custom_env, tmpdir
):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(
        sk_model=sklearn_knn_model,
        path=sklearn_model_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(predict_fn=None),
        conda_env=pyfunc_custom_env,
    )

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )
    saved_conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pyfunc_custom_env

    with open(pyfunc_custom_env, "r") as f:
        pyfunc_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pyfunc_custom_env_parsed


@pytest.mark.large
def test_save_model_persists_requirements_in_mlflow_model_directory(
    sklearn_knn_model, main_scoped_model_class, pyfunc_custom_env, tmpdir
):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(
        sk_model=sklearn_knn_model,
        path=sklearn_model_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(predict_fn=None),
        conda_env=pyfunc_custom_env,
    )

    saved_pip_req_path = os.path.join(pyfunc_model_path, "requirements.txt")
    _compare_conda_env_requirements(pyfunc_custom_env, saved_pip_req_path)


@pytest.mark.large
def test_log_model_with_pip_requirements(main_scoped_model_class, tmpdir):
    python_model = main_scoped_model_class(predict_fn=None)
    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            "model", python_model=python_model, pip_requirements=req_file.strpath
        )
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", "a"], strict=True)

    # List of requirements
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            "model", python_model=python_model, pip_requirements=[f"-r {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            "model", python_model=python_model, pip_requirements=[f"-c {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            ["mlflow", "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


@pytest.mark.large
def test_log_model_with_extra_pip_requirements(sklearn_knn_model, main_scoped_model_class, tmpdir):
    sklearn_model_path = tmpdir.join("sklearn_model").strpath
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    python_model = main_scoped_model_class(predict_fn=None)
    default_reqs = mlflow.pyfunc.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            "model",
            python_model=python_model,
            artifacts={"sk_model": sklearn_model_path},
            extra_pip_requirements=req_file.strpath,
        )
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a"])

    # List of requirements
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            "model",
            artifacts={"sk_model": sklearn_model_path},
            python_model=python_model,
            extra_pip_requirements=[f"-r {req_file.strpath}", "b"],
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            "model",
            artifacts={"sk_model": sklearn_model_path},
            python_model=python_model,
            extra_pip_requirements=[f"-c {req_file.strpath}", "b"],
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            ["mlflow", *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


@pytest.mark.large
def test_log_model_persists_specified_conda_env_in_mlflow_model_directory(
    sklearn_knn_model, main_scoped_model_class, pyfunc_custom_env
):
    sklearn_artifact_path = "sk_model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_knn_model, artifact_path=sklearn_artifact_path)
        sklearn_run_id = mlflow.active_run().info.run_id

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            artifacts={
                "sk_model": utils_get_artifact_uri(
                    artifact_path=sklearn_artifact_path, run_id=sklearn_run_id
                )
            },
            python_model=main_scoped_model_class(predict_fn=None),
            conda_env=pyfunc_custom_env,
        )
        pyfunc_model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=pyfunc_artifact_path
            )
        )

    pyfunc_conf = _get_flavor_configuration(
        model_path=pyfunc_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )
    saved_conda_env_path = os.path.join(pyfunc_model_path, pyfunc_conf[mlflow.pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pyfunc_custom_env

    with open(pyfunc_custom_env, "r") as f:
        pyfunc_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pyfunc_custom_env_parsed


@pytest.mark.large
def test_model_log_persists_requirements_in_mlflow_model_directory(
    sklearn_knn_model, main_scoped_model_class, pyfunc_custom_env
):
    sklearn_artifact_path = "sk_model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_knn_model, artifact_path=sklearn_artifact_path)
        sklearn_run_id = mlflow.active_run().info.run_id

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            artifacts={
                "sk_model": utils_get_artifact_uri(
                    artifact_path=sklearn_artifact_path, run_id=sklearn_run_id
                )
            },
            python_model=main_scoped_model_class(predict_fn=None),
            conda_env=pyfunc_custom_env,
        )
        pyfunc_model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=pyfunc_artifact_path
            )
        )

    saved_pip_req_path = os.path.join(pyfunc_model_path, "requirements.txt")
    _compare_conda_env_requirements(pyfunc_custom_env, saved_pip_req_path)


@pytest.mark.large
def test_save_model_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    sklearn_logreg_model, main_scoped_model_class, tmpdir
):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_model")
    mlflow.sklearn.save_model(sk_model=sklearn_logreg_model, path=sklearn_model_path)

    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_model")
    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        python_model=main_scoped_model_class(predict_fn=None),
        conda_env=_conda_env(),
    )
    _assert_pip_requirements(pyfunc_model_path, mlflow.pyfunc.get_default_pip_requirements())


@pytest.mark.large
def test_log_model_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    sklearn_knn_model, main_scoped_model_class
):
    sklearn_artifact_path = "sk_model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_knn_model, artifact_path=sklearn_artifact_path)
        sklearn_run_id = mlflow.active_run().info.run_id

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            artifacts={
                "sk_model": utils_get_artifact_uri(
                    artifact_path=sklearn_artifact_path, run_id=sklearn_run_id
                )
            },
            python_model=main_scoped_model_class(predict_fn=None),
        )
        model_uri = mlflow.get_artifact_uri(pyfunc_artifact_path)
    _assert_pip_requirements(model_uri, mlflow.pyfunc.get_default_pip_requirements())


@pytest.mark.large
def test_save_model_correctly_resolves_directory_artifact_with_nested_contents(
    tmpdir, model_path, iris_data
):
    directory_artifact_path = os.path.join(str(tmpdir), "directory_artifact")
    nested_file_relative_path = os.path.join(
        "my", "somewhat", "heavily", "nested", "directory", "myfile.txt"
    )
    nested_file_path = os.path.join(directory_artifact_path, nested_file_relative_path)
    os.makedirs(os.path.dirname(nested_file_path))
    nested_file_text = "some sample file text"
    with open(nested_file_path, "w") as f:
        f.write(nested_file_text)

    class ArtifactValidationModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            expected_file_path = os.path.join(
                context.artifacts["testdir"], nested_file_relative_path
            )
            if not os.path.exists(expected_file_path):
                return False
            else:
                with open(expected_file_path, "r") as f:
                    return f.read() == nested_file_text

    mlflow.pyfunc.save_model(
        path=model_path,
        artifacts={"testdir": directory_artifact_path},
        python_model=ArtifactValidationModel(),
        conda_env=_conda_env(),
    )

    loaded_model = mlflow.pyfunc.load_pyfunc(model_uri=model_path)
    assert loaded_model.predict(iris_data[0])


@pytest.mark.large
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


@pytest.mark.large
def test_save_model_with_python_model_argument_of_invalid_type_raises_exeption(tmpdir):
    match = "python_model` must be a subclass of `PythonModel`"
    with pytest.raises(MlflowException, match=match):
        mlflow.pyfunc.save_model(
            path=os.path.join(str(tmpdir), "model1"), python_model="not the right type"
        )

    with pytest.raises(MlflowException, match=match):
        mlflow.pyfunc.save_model(
            path=os.path.join(str(tmpdir), "model2"), python_model="not the right type"
        )


@pytest.mark.large
def test_save_model_with_unsupported_argument_combinations_throws_exception(model_path):
    with pytest.raises(
        MlflowException, match="Either `loader_module` or `python_model` must be specified"
    ) as exc_info:
        mlflow.pyfunc.save_model(
            path=model_path, artifacts={"artifact": "/path/to/artifact"}, python_model=None
        )

    python_model = ModuleScopedSklearnModel(predict_fn=None)
    loader_module = __name__
    with pytest.raises(
        MlflowException, match="The following sets of parameters cannot be specified together"
    ) as exc_info:
        mlflow.pyfunc.save_model(
            path=model_path, python_model=python_model, loader_module=loader_module
        )
    assert str(python_model) in str(exc_info)
    assert str(loader_module) in str(exc_info)

    with pytest.raises(
        MlflowException, match="The following sets of parameters cannot be specified together"
    ) as exc_info:
        mlflow.pyfunc.save_model(
            path=model_path,
            python_model=python_model,
            data_path="/path/to/data",
            artifacts={"artifact": "/path/to/artifact"},
        )

    with pytest.raises(
        MlflowException, match="Either `loader_module` or `python_model` must be specified"
    ):
        mlflow.pyfunc.save_model(path=model_path, python_model=None, loader_module=None)


@pytest.mark.large
def test_log_model_with_unsupported_argument_combinations_throws_exception():
    match = (
        "Either `loader_module` or `python_model` must be specified. A `loader_module` "
        "should be a python module. A `python_model` should be a subclass of "
        "PythonModel"
    )
    with mlflow.start_run(), pytest.raises(MlflowException, match=match):
        mlflow.pyfunc.log_model(
            artifact_path="pyfunc_model",
            artifacts={"artifact": "/path/to/artifact"},
            python_model=None,
        )

    python_model = ModuleScopedSklearnModel(predict_fn=None)
    loader_module = __name__
    with mlflow.start_run(), pytest.raises(
        MlflowException,
        match="The following sets of parameters cannot be specified together",
    ) as exc_info:
        mlflow.pyfunc.log_model(
            artifact_path="pyfunc_model",
            python_model=python_model,
            loader_module=loader_module,
        )
    assert str(python_model) in str(exc_info)
    assert str(loader_module) in str(exc_info)

    with mlflow.start_run(), pytest.raises(
        MlflowException, match="The following sets of parameters cannot be specified together"
    ) as exc_info:
        mlflow.pyfunc.log_model(
            artifact_path="pyfunc_model",
            python_model=python_model,
            data_path="/path/to/data",
            artifacts={"artifact1": "/path/to/artifact"},
        )

    with mlflow.start_run(), pytest.raises(
        MlflowException, match="Either `loader_module` or `python_model` must be specified"
    ):
        mlflow.pyfunc.log_model(artifact_path="pyfunc_model", python_model=None, loader_module=None)


@pytest.mark.large
def test_repr_can_be_called_withtout_run_id_or_artifact_path():
    model_meta = Model(
        artifact_path=None,
        run_id=None,
        flavors={"python_function": {"loader_module": "someFlavour"}},
    )

    class TestModel:
        def predict(self, model_input):
            return model_input

    model_impl = TestModel()

    assert "flavor: someFlavour" in mlflow.pyfunc.PyFuncModel(model_meta, model_impl).__repr__()


@pytest.mark.large
def test_load_model_with_differing_cloudpickle_version_at_micro_granularity_logs_warning(
    model_path,
):
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
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
    with mock.patch("mlflow.pyfunc._logger.warning") as warn_mock, mock.patch(
        "cloudpickle.__version__"
    ) as cloudpickle_version_mock:
        cloudpickle_version_mock.__str__ = lambda *args, **kwargs: loader_cloudpickle_version
        warn_mock.side_effect = custom_warn
        mlflow.pyfunc.load_pyfunc(model_uri=model_path)

    assert any(
        [
            "differs from the version of CloudPickle that is currently running" in log_message
            and saver_cloudpickle_version in log_message
            and loader_cloudpickle_version in log_message
            for log_message in log_messages
        ]
    )


@pytest.mark.large
def test_load_model_with_missing_cloudpickle_version_logs_warning(model_path):
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
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
        mlflow.pyfunc.load_pyfunc(model_uri=model_path)

    assert any(
        [
            (
                "The version of CloudPickle used to save the model could not be found"
                " in the MLmodel configuration"
            )
            in log_message
            for log_message in log_messages
        ]
    )


@pytest.mark.large
def test_save_and_load_model_with_special_chars(
    sklearn_knn_model, main_scoped_model_class, iris_data, tmpdir
):
    sklearn_model_path = os.path.join(str(tmpdir), "sklearn_  model")
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    def test_predict(sk_model, model_input):
        return sk_model.predict(model_input) * 2

    # Intentionally create a path that has non-url-compatible characters
    pyfunc_model_path = os.path.join(str(tmpdir), "pyfunc_ :% model")

    mlflow.pyfunc.save_model(
        path=pyfunc_model_path,
        artifacts={"sk_model": sklearn_model_path},
        conda_env=_conda_env(),
        python_model=main_scoped_model_class(test_predict),
    )

    loaded_pyfunc_model = mlflow.pyfunc.load_pyfunc(model_uri=pyfunc_model_path)
    np.testing.assert_array_equal(
        loaded_pyfunc_model.predict(iris_data[0]),
        test_predict(sk_model=sklearn_knn_model, model_input=iris_data[0]),
    )
