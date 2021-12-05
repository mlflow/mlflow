import sys
import os
import pytest
import numpy as np
from unittest import mock
from unittest.mock import Mock

import pandas as pd
import pandas.testing
import sklearn.datasets as datasets
from sklearn.linear_model import LogisticRegression


import mlflow
import mlflow.azureml
import mlflow.azureml.cli
import mlflow.sklearn
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.utils.file_utils import TempDir

from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import

pytestmark = pytest.mark.skipif(
    (sys.version_info < (3, 0)), reason="Tests require Python 3 to run!"
)


class AzureMLMocks:
    def __init__(self):
        self.mocks = {
            "register_model": mock.patch("azureml.core.model.Model.register"),
            "get_model_path": mock.patch("azureml.core.model.Model.get_model_path"),
            "model_deploy": mock.patch("azureml.core.model.Model.deploy"),
            "load_workspace": mock.patch("azureml.core.Workspace.get"),
        }

    def __getitem__(self, key):
        return self.mocks[key]

    def __enter__(self):
        for key, mock in self.mocks.items():
            self.mocks[key] = mock.__enter__()
        return self

    def __exit__(self, *args):
        for mock in self.mocks.values():
            mock.__exit__(*args)


def get_azure_workspace():
    # pylint: disable=import-error
    from azureml.core import Workspace

    return Workspace.get("test_workspace")


@pytest.fixture(scope="module")
def sklearn_data():
    iris = datasets.load_iris()
    x = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    return x, y


@pytest.fixture(scope="module")
def sklearn_model(sklearn_data):
    x, y = sklearn_data
    linear_lr = LogisticRegression()
    linear_lr.fit(x, y)
    return linear_lr


class LogisticRegressionPandas(LogisticRegression):
    def predict(self, *args, **kwargs):  # pylint: disable=arguments-differ
        # Wrap the output with `pandas.DataFrame`
        return pd.DataFrame(super().predict(*args, **kwargs))


@pytest.fixture(scope="module")
def sklearn_pd_model(sklearn_data):
    x, y = sklearn_data
    linear_lr = LogisticRegressionPandas()
    linear_lr.fit(x, y)
    return linear_lr


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_deploy_with_absolute_model_path_calls_expected_azure_routines(sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks() as aml_mocks:
        workspace = get_azure_workspace()
        mlflow.azureml.deploy(model_uri=model_path, workspace=workspace)

        assert aml_mocks["register_model"].call_count == 1
        assert aml_mocks["model_deploy"].call_count == 1


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_deploy_with_relative_model_path_calls_expected_azure_routines(sklearn_model):
    with TempDir(chdr=True):
        model_path = "model"
        mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
        with AzureMLMocks() as aml_mocks:
            workspace = get_azure_workspace()
            mlflow.azureml.deploy(model_uri=model_path, workspace=workspace)

            assert aml_mocks["register_model"].call_count == 1
            assert aml_mocks["model_deploy"].call_count == 1


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_deploy_with_runs_uri_calls_expected_azure_routines(sklearn_model):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_model, artifact_path=artifact_path)
        run_id = mlflow.active_run().info.run_id

    with AzureMLMocks() as aml_mocks:
        workspace = get_azure_workspace()
        model_uri = "runs:///{run_id}/{artifact_path}".format(
            run_id=run_id, artifact_path=artifact_path
        )
        mlflow.azureml.deploy(model_uri=model_uri, workspace=workspace)

        assert aml_mocks["register_model"].call_count == 1
        assert aml_mocks["model_deploy"].call_count == 1


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_deploy_with_remote_uri_calls_expected_azure_routines(
    sklearn_model, model_path, mock_s3_bucket
):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    artifact_path = "model"
    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    s3_artifact_repo = S3ArtifactRepository(artifact_root)
    s3_artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)
    model_uri = artifact_root + "/" + artifact_path

    with AzureMLMocks() as aml_mocks:
        workspace = get_azure_workspace()
        mlflow.azureml.deploy(model_uri=model_uri, workspace=workspace)

        assert aml_mocks["register_model"].call_count == 1
        assert aml_mocks["model_deploy"].call_count == 1


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_synchronous_deploy_awaits_azure_service_creation(sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks():
        workspace = get_azure_workspace()
        service, _ = mlflow.azureml.deploy(
            model_uri=model_path, workspace=workspace, synchronous=True
        )
        service.wait_for_deployment.assert_called_once()


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_asynchronous_deploy_does_not_await_azure_service_creation(sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks():
        workspace = get_azure_workspace()
        service, _ = mlflow.azureml.deploy(
            model_uri=model_path, workspace=workspace, synchronous=False
        )
        service.wait_for_deployment.assert_not_called()


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_deploy_registers_model_and_creates_service_with_specified_names(sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks() as aml_mocks:
        workspace = get_azure_workspace()
        model_name = "MODEL_NAME_1"
        service_name = "service_name_1"
        mlflow.azureml.deploy(
            model_uri=model_path,
            workspace=workspace,
            model_name=model_name,
            service_name=service_name,
        )

        register_model_call_args = aml_mocks["register_model"].call_args_list
        assert len(register_model_call_args) == 1
        _, register_model_call_kwargs = register_model_call_args[0]
        assert register_model_call_kwargs["model_name"] == model_name

        model_deploy_call_args = aml_mocks["model_deploy"].call_args_list
        assert len(model_deploy_call_args) == 1
        _, model_deploy_call_kwargs = model_deploy_call_args[0]
        assert model_deploy_call_kwargs["name"] == service_name


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_deploy_generates_model_and_service_names_meeting_azureml_resource_naming_requirements(
    sklearn_model, model_path
):
    aml_resource_name_max_length = 32

    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks() as aml_mocks:
        workspace = get_azure_workspace()
        mlflow.azureml.deploy(model_uri=model_path, workspace=workspace)

        register_model_call_args = aml_mocks["register_model"].call_args_list
        assert len(register_model_call_args) == 1
        _, register_model_call_kwargs = register_model_call_args[0]
        called_model_name = register_model_call_kwargs["model_name"]
        assert len(called_model_name) <= aml_resource_name_max_length

        model_deploy_call_args = aml_mocks["model_deploy"].call_args_list
        assert len(model_deploy_call_args) == 1
        _, model_deploy_call_kwargs = model_deploy_call_args[0]
        called_service_name = model_deploy_call_kwargs["name"]
        assert len(called_service_name) <= aml_resource_name_max_length


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_deploy_passes_model_conda_environment_to_azure_service_creation_routine(
    sklearn_model, model_path
):
    sklearn_conda_env_text = """\
    name: sklearn-env
    dependencies:
        - scikit-learn
    """
    with TempDir(chdr=True) as tmp:
        sklearn_conda_env_path = tmp.path("conda.yaml")
        with open(sklearn_conda_env_path, "w") as f:
            f.write(sklearn_conda_env_text)

        mlflow.sklearn.save_model(
            sk_model=sklearn_model, path=model_path, conda_env=sklearn_conda_env_path
        )

        # Mock the TempDir.__exit__ function to ensure that the enclosing temporary
        # directory is not deleted
        with AzureMLMocks() as aml_mocks, mock.patch(
            "mlflow.utils.file_utils.TempDir.path"
        ) as tmpdir_path_mock, mock.patch("mlflow.utils.file_utils.TempDir.__exit__"):

            def get_mock_path(subpath):
                # Our current working directory is a temporary directory. Therefore, it is safe to
                # directly return the specified subpath.
                return subpath

            tmpdir_path_mock.side_effect = get_mock_path

            workspace = get_azure_workspace()
            mlflow.azureml.deploy(model_uri=model_path, workspace=workspace)

            model_deploy_call_args = aml_mocks["model_deploy"].call_args_list
            assert len(model_deploy_call_args) == 1
            _, model_deploy_call_kwargs = model_deploy_call_args[0]
            service_config = model_deploy_call_kwargs["inference_config"]
            conda_deps = service_config.environment.python.conda_dependencies
            assert conda_deps is not None
            assert "scikit-learn" in conda_deps.conda_packages


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_deploy_throws_exception_if_model_does_not_contain_pyfunc_flavor(sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    model_config_path = os.path.join(model_path, "MLmodel")
    model_config = Model.load(model_config_path)
    del model_config.flavors[pyfunc.FLAVOR_NAME]
    model_config.save(model_config_path)

    with AzureMLMocks(), pytest.raises(
        MlflowException, match="does not contain the `python_function` flavor"
    ) as exc:
        workspace = get_azure_workspace()
        mlflow.azureml.deploy(model_uri=model_path, workspace=workspace)
        assert exc.error_code == INVALID_PARAMETER_VALUE


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_deploy_throws_exception_if_model_python_version_is_less_than_three(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    model_config_path = os.path.join(model_path, "MLmodel")
    model_config = Model.load(model_config_path)
    model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.PY_VERSION] = "2.7.6"
    model_config.save(model_config_path)

    with AzureMLMocks(), pytest.raises(MlflowException, match="Python 3 and above") as exc:
        workspace = get_azure_workspace()
        mlflow.azureml.deploy(model_uri=model_path, workspace=workspace)
        assert exc.error_code == INVALID_PARAMETER_VALUE


@pytest.mark.large
def test_execution_script_init_method_attempts_to_load_correct_azure_ml_model(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)

    model_name = "test_model_name"
    model_version = 1

    model_mock = Mock()
    model_mock.name = model_name
    model_mock.version = model_version

    with TempDir() as tmp:
        execution_script_path = tmp.path("dest")
        mlflow.azureml._create_execution_script(
            output_path=execution_script_path, azure_model=model_mock
        )

        with open(execution_script_path, "r") as f:
            execution_script = f.read()

    # Define the `init` and `score` methods contained in the execution script
    # pylint: disable=exec-used
    # Define an empty globals dictionary to ensure that the initialize of the execution
    # script does not depend on the current state of the test environment
    globs = {}
    exec(execution_script, globs)
    # Update the set of global variables available to the test environment to include
    # functions defined during the evaluation of the execution script
    globals().update(globs)
    with AzureMLMocks() as aml_mocks:
        aml_mocks["get_model_path"].side_effect = lambda *args, **kwargs: model_path
        # Execute the `init` method of the execution script.
        # pylint: disable=undefined-variable
        init()

        assert aml_mocks["get_model_path"].call_count == 1
        get_model_path_call_args = aml_mocks["get_model_path"].call_args_list
        assert len(get_model_path_call_args) == 1
        _, get_model_path_call_kwargs = get_model_path_call_args[0]
        assert get_model_path_call_kwargs["model_name"] == model_name
        assert get_model_path_call_kwargs["version"] == model_version


@pytest.mark.large
def test_execution_script_run_method_scores_pandas_dfs_successfully_when_model_outputs_numpy_arrays(
    sklearn_model, sklearn_data, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)

    pyfunc_model = mlflow.pyfunc.load_pyfunc(model_uri=model_path)
    pyfunc_outputs = pyfunc_model.predict(sklearn_data[0])
    assert isinstance(pyfunc_outputs, np.ndarray)

    model_mock = Mock()
    model_mock.name = "model_name"
    model_mock.version = 1

    with TempDir() as tmp:
        execution_script_path = tmp.path("dest")
        mlflow.azureml._create_execution_script(
            output_path=execution_script_path, azure_model=model_mock
        )

        with open(execution_script_path, "r") as f:
            execution_script = f.read()

    # Define the `init` and `score` methods contained in the execution script
    # pylint: disable=exec-used
    # Define an empty globals dictionary to ensure that the initialize of the execution
    # script does not depend on the current state of the test environment
    globs = {}
    exec(execution_script, globs)
    # Update the set of global variables available to the test environment to include
    # functions defined during the evaluation of the execution script
    globals().update(globs)
    with AzureMLMocks() as aml_mocks:
        aml_mocks["get_model_path"].side_effect = lambda *args, **kwargs: model_path
        # Execute the `init` method of the execution script and load the sklearn model from the
        # mocked path
        # pylint: disable=undefined-variable
        init()

        # Invoke the `run` method of the execution script with sample input data and verify that
        # reasonable output data is produced
        # pylint: disable=undefined-variable
        output_data = run(pd.DataFrame(data=sklearn_data[0]).to_json(orient="split"))
        np.testing.assert_array_equal(output_data, pyfunc_outputs)


@pytest.mark.large
def test_execution_script_run_method_scores_pandas_dfs_successfully_when_model_outputs_pandas_dfs(
    sklearn_pd_model, sklearn_data, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_pd_model, path=model_path)
    pyfunc_model = mlflow.pyfunc.load_pyfunc(model_uri=model_path)
    pyfunc_outputs = pyfunc_model.predict(sklearn_data[0])
    assert isinstance(pyfunc_outputs, pd.DataFrame)

    model_mock = Mock()
    model_mock.name = "model_name"
    model_mock.version = 1

    with TempDir() as tmp:
        execution_script_path = tmp.path("dest")
        mlflow.azureml._create_execution_script(
            output_path=execution_script_path, azure_model=model_mock
        )

        with open(execution_script_path, "r") as f:
            execution_script = f.read()

    # Define the `init` and `score` methods contained in the execution script
    # pylint: disable=exec-used
    # Define an empty globals dictionary to ensure that the initialize of the execution
    # script does not depend on the current state of the test environment
    globs = {}
    exec(execution_script, globs)
    # Update the set of global variables available to the test environment to include
    # functions defined during the evaluation of the execution script
    globals().update(globs)
    with AzureMLMocks() as aml_mocks:
        aml_mocks["get_model_path"].side_effect = lambda *args, **kwargs: model_path
        # Execute the `init` method of the execution script and load the sklearn model from the
        # mocked path
        # pylint: disable=undefined-variable
        init()

        # Invoke the `run` method of the execution script with sample input data and verify that
        # reasonable output data is produced
        # pylint: disable=undefined-variable
        output_raw = run(pd.DataFrame(data=sklearn_data[0]).to_json(orient="split"))
        output_df = pd.DataFrame(output_raw)
        pandas.testing.assert_frame_equal(
            output_df, pyfunc_outputs, check_dtype=False, check_less_precise=False
        )
