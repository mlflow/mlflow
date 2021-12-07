import sys
import os
import json
import pytest
import yaml
import numpy as np
from unittest import mock
from unittest.mock import Mock

import pandas as pd
import pandas.testing
import sklearn.datasets as datasets
from sklearn.linear_model import LogisticRegression
from click.testing import CliRunner

import mlflow
import mlflow.azureml
import mlflow.azureml.cli
import mlflow.sklearn
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
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
            "create_image": mock.patch("azureml.core.Image.create"),
            "deploy": mock.patch("azureml.core.model.Model.deploy"),
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
def test_build_image_with_absolute_model_path_calls_expected_azure_routines(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks() as aml_mocks:
        workspace = get_azure_workspace()
        mlflow.azureml.build_image(model_uri=model_path, workspace=workspace)

        assert aml_mocks["register_model"].call_count == 1
        assert aml_mocks["create_image"].call_count == 1


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_build_image_with_relative_model_path_calls_expected_azure_routines(sklearn_model):
    with TempDir(chdr=True):
        model_path = "model"
        mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
        with AzureMLMocks() as aml_mocks:
            workspace = get_azure_workspace()
            mlflow.azureml.build_image(model_uri=model_path, workspace=workspace)

            assert aml_mocks["register_model"].call_count == 1
            assert aml_mocks["create_image"].call_count == 1


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_build_image_with_runs_uri_calls_expected_azure_routines(sklearn_model):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_model, artifact_path=artifact_path)
        run_id = mlflow.active_run().info.run_id

    with AzureMLMocks() as aml_mocks:
        workspace = get_azure_workspace()
        model_uri = "runs:///{run_id}/{artifact_path}".format(
            run_id=run_id, artifact_path=artifact_path
        )
        mlflow.azureml.build_image(model_uri=model_uri, workspace=workspace)

        assert aml_mocks["register_model"].call_count == 1
        assert aml_mocks["create_image"].call_count == 1


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_build_image_with_remote_uri_calls_expected_azure_routines(
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
        mlflow.azureml.build_image(model_uri=model_uri, workspace=workspace)

        assert aml_mocks["register_model"].call_count == 1
        assert aml_mocks["create_image"].call_count == 1


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_synchronous_build_image_awaits_azure_image_creation(sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks():
        workspace = get_azure_workspace()
        image, _ = mlflow.azureml.build_image(
            model_uri=model_path, workspace=workspace, synchronous=True
        )
        image.wait_for_creation.assert_called_once()


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_asynchronous_build_image_does_not_await_azure_image_creation(sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks():
        workspace = get_azure_workspace()
        image, _ = mlflow.azureml.build_image(
            model_uri=model_path, workspace=workspace, synchronous=False
        )
        image.wait_for_creation.assert_not_called()


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_build_image_registers_model_and_creates_image_with_specified_names(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks() as aml_mocks:
        workspace = get_azure_workspace()
        model_name = "MODEL_NAME_1"
        image_name = "IMAGE_NAME_1"
        mlflow.azureml.build_image(
            model_uri=model_path, workspace=workspace, model_name=model_name, image_name=image_name
        )

        register_model_call_args = aml_mocks["register_model"].call_args_list
        assert len(register_model_call_args) == 1
        _, register_model_call_kwargs = register_model_call_args[0]
        assert register_model_call_kwargs["model_name"] == model_name

        create_image_call_args = aml_mocks["create_image"].call_args_list
        assert len(create_image_call_args) == 1
        _, create_image_call_kwargs = create_image_call_args[0]
        assert create_image_call_kwargs["name"] == image_name


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_build_image_generates_model_and_image_names_meeting_azureml_resource_naming_requirements(
    sklearn_model, model_path
):
    aml_resource_name_max_length = 32

    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks() as aml_mocks:
        workspace = get_azure_workspace()
        mlflow.azureml.build_image(model_uri=model_path, workspace=workspace)

        register_model_call_args = aml_mocks["register_model"].call_args_list
        assert len(register_model_call_args) == 1
        _, register_model_call_kwargs = register_model_call_args[0]
        called_model_name = register_model_call_kwargs["model_name"]
        assert len(called_model_name) <= aml_resource_name_max_length

        create_image_call_args = aml_mocks["create_image"].call_args_list
        assert len(create_image_call_args) == 1
        _, create_image_call_kwargs = create_image_call_args[0]
        called_image_name = create_image_call_kwargs["name"]
        assert len(called_image_name) <= aml_resource_name_max_length


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_build_image_passes_model_conda_environment_to_azure_image_creation_routine(
    sklearn_model, model_path
):
    sklearn_conda_env_text = """\
    name: sklearn-env
    dependencies:
        - pip:
          - mlflow
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
            mlflow.azureml.build_image(model_uri=model_path, workspace=workspace)

            create_image_call_args = aml_mocks["create_image"].call_args_list
            assert len(create_image_call_args) == 1
            _, create_image_call_kwargs = create_image_call_args[0]
            image_config = create_image_call_kwargs["image_config"]
            assert image_config.conda_file is not None
            with open(image_config.conda_file, "r") as f:
                assert yaml.safe_load(f.read()) == yaml.safe_load(sklearn_conda_env_text)


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_build_image_includes_default_metadata_in_azure_image_and_model_tags(sklearn_model):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_model, artifact_path=artifact_path)
        run_id = mlflow.active_run().info.run_id
    model_uri = "runs:///{run_id}/{artifact_path}".format(
        run_id=run_id, artifact_path=artifact_path
    )
    model_config = Model.load(
        os.path.join(_download_artifact_from_uri(artifact_uri=model_uri), "MLmodel")
    )

    with AzureMLMocks() as aml_mocks:
        workspace = get_azure_workspace()
        mlflow.azureml.build_image(model_uri=model_uri, workspace=workspace)

        register_model_call_args = aml_mocks["register_model"].call_args_list
        assert len(register_model_call_args) == 1
        _, register_model_call_kwargs = register_model_call_args[0]
        called_tags = register_model_call_kwargs["tags"]
        assert called_tags["model_uri"] == model_uri
        assert (
            called_tags["python_version"]
            == model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.PY_VERSION]
        )

        create_image_call_args = aml_mocks["create_image"].call_args_list
        assert len(create_image_call_args) == 1
        _, create_image_call_kwargs = create_image_call_args[0]
        image_config = create_image_call_kwargs["image_config"]
        assert image_config.tags["model_uri"] == model_uri
        assert (
            image_config.tags["python_version"]
            == model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.PY_VERSION]
        )


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_build_image_includes_user_specified_tags_in_azure_image_and_model_tags(
    sklearn_model, model_path
):
    custom_tags = {
        "User": "Corey",
        "Date": "Today",
        "Other": "Entry",
    }

    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks() as aml_mocks:
        workspace = get_azure_workspace()
        mlflow.azureml.build_image(model_uri=model_path, workspace=workspace, tags=custom_tags)

        register_model_call_args = aml_mocks["register_model"].call_args_list
        assert len(register_model_call_args) == 1
        _, register_model_call_kwargs = register_model_call_args[0]
        called_tags = register_model_call_kwargs["tags"]
        assert custom_tags.items() <= called_tags.items()

        create_image_call_args = aml_mocks["create_image"].call_args_list
        assert len(create_image_call_args) == 1
        _, create_image_call_kwargs = create_image_call_args[0]
        image_config = create_image_call_kwargs["image_config"]
        assert custom_tags.items() <= image_config.tags.items()


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_deploy_includes_tags_in_azure_deployment_and_model_tags(sklearn_model, model_path):
    custom_tags = {
        "User": "Corey",
        "Date": "Today",
        "Other": "Entry",
    }

    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks() as aml_mocks:
        workspace = get_azure_workspace()
        mlflow.azureml.deploy(model_uri=model_path, workspace=workspace, tags=custom_tags)

        register_model_call_args = aml_mocks["register_model"].call_args_list
        assert len(register_model_call_args) == 1
        _, register_model_call_kwargs = register_model_call_args[0]
        called_tags = register_model_call_kwargs["tags"]
        assert custom_tags.items() <= called_tags.items()

        deploy_call_args = aml_mocks["deploy"].call_args_list
        assert len(deploy_call_args) == 1
        _, deploy_call_kwargs = deploy_call_args[0]
        deployment_config = deploy_call_kwargs["deployment_config"]
        assert custom_tags.items() <= deployment_config.tags.items()


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_build_image_includes_user_specified_description_in_azure_image_and_model_tags(
    sklearn_model, model_path
):
    custom_description = "a custom description"

    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks() as aml_mocks:
        workspace = get_azure_workspace()
        mlflow.azureml.build_image(
            model_uri=model_path, workspace=workspace, description=custom_description
        )

        register_model_call_args = aml_mocks["register_model"].call_args_list
        assert len(register_model_call_args) == 1
        _, register_model_call_kwargs = register_model_call_args[0]
        assert register_model_call_kwargs["description"] == custom_description

        create_image_call_args = aml_mocks["create_image"].call_args_list
        assert len(create_image_call_args) == 1
        _, create_image_call_kwargs = create_image_call_args[0]
        image_config = create_image_call_kwargs["image_config"]
        assert image_config.description == custom_description


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_build_image_throws_exception_if_model_does_not_contain_pyfunc_flavor(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    model_config_path = os.path.join(model_path, "MLmodel")
    model_config = Model.load(model_config_path)
    del model_config.flavors[pyfunc.FLAVOR_NAME]
    model_config.save(model_config_path)

    with AzureMLMocks(), pytest.raises(
        MlflowException, match="does not contain the `python_function` flavor"
    ) as exc:
        workspace = get_azure_workspace()
        mlflow.azureml.build_image(model_uri=model_path, workspace=workspace)
        assert exc.error_code == INVALID_PARAMETER_VALUE


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_build_image_throws_exception_if_model_python_version_is_less_than_three(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    model_config_path = os.path.join(model_path, "MLmodel")
    model_config = Model.load(model_config_path)
    model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.PY_VERSION] = "2.7.6"
    model_config.save(model_config_path)

    with AzureMLMocks(), pytest.raises(MlflowException, match="Python 3 and above") as exc:
        workspace = get_azure_workspace()
        mlflow.azureml.build_image(model_uri=model_path, workspace=workspace)
        assert exc.error_code == INVALID_PARAMETER_VALUE


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_build_image_includes_mlflow_home_as_file_dependency_if_specified(
    sklearn_model, model_path
):
    def mock_create_dockerfile(output_path, *args, **kwargs):
        # pylint: disable=unused-argument
        with open(output_path, "w") as f:
            f.write("Dockerfile contents")

    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks() as aml_mocks, TempDir() as tmp, mock.patch(
        "mlflow.azureml._create_dockerfile"
    ) as create_dockerfile_mock:
        create_dockerfile_mock.side_effect = mock_create_dockerfile

        # Write a mock `setup.py` file to the mlflow home path so that it will be recognized
        # as a viable MLflow source directory during the image build process
        mlflow_home = tmp.path()
        with open(os.path.join(mlflow_home, "setup.py"), "w") as f:
            f.write("setup instructions")

        workspace = get_azure_workspace()
        mlflow.azureml.build_image(
            model_uri=model_path, workspace=workspace, mlflow_home=mlflow_home
        )

        assert len(create_dockerfile_mock.call_args_list) == 1
        _, create_dockerfile_kwargs = create_dockerfile_mock.call_args_list[0]
        # The path to MLflow that is referenced by the Docker container may differ from the
        # user-specified `mlflow_home` path if the directory is copied before image building
        # for safety
        dockerfile_mlflow_path = create_dockerfile_kwargs["mlflow_path"]

        create_image_call_args = aml_mocks["create_image"].call_args_list
        assert len(create_image_call_args) == 1
        _, create_image_call_kwargs = create_image_call_args[0]
        image_config = create_image_call_kwargs["image_config"]
        assert dockerfile_mlflow_path in image_config.dependencies


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


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_cli_build_image_with_absolute_model_path_calls_expected_azure_routines(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with AzureMLMocks() as aml_mocks:
        result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
            mlflow.azureml.cli.commands,
            [
                "build-image",
                "-m",
                model_path,
                "-w",
                "test_workspace",
                "-i",
                "image_name",
                "-n",
                "model_name",
            ],
        )
        assert result.exit_code == 0

        assert aml_mocks["register_model"].call_count == 1
        assert aml_mocks["create_image"].call_count == 1
        assert aml_mocks["load_workspace"].call_count == 1


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_cli_build_image_with_relative_model_path_calls_expected_azure_routines(sklearn_model):
    with TempDir(chdr=True):
        model_path = "model"
        mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)

        with AzureMLMocks() as aml_mocks:
            result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
                mlflow.azureml.cli.commands,
                [
                    "build-image",
                    "-m",
                    model_path,
                    "-w",
                    "test_workspace",
                    "-i",
                    "image_name",
                    "-n",
                    "model_name",
                ],
            )
            assert result.exit_code == 0

            assert aml_mocks["register_model"].call_count == 1
            assert aml_mocks["create_image"].call_count == 1
            assert aml_mocks["load_workspace"].call_count == 1


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_cli_build_image_with_runs_uri_calls_expected_azure_routines(sklearn_model):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_model, artifact_path=artifact_path)
        run_id = mlflow.active_run().info.run_id
    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)

    with AzureMLMocks() as aml_mocks:
        result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
            mlflow.azureml.cli.commands,
            [
                "build-image",
                "-m",
                model_uri,
                "-w",
                "test_workspace",
                "-i",
                "image_name",
                "-n",
                "model_name",
            ],
        )
        assert result.exit_code == 0

        assert aml_mocks["register_model"].call_count == 1
        assert aml_mocks["create_image"].call_count == 1
        assert aml_mocks["load_workspace"].call_count == 1


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_cli_build_image_with_remote_uri_calls_expected_azure_routines(
    sklearn_model, model_path, mock_s3_bucket
):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    artifact_path = "model"
    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    s3_artifact_repo = S3ArtifactRepository(artifact_root)
    s3_artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)
    model_uri = artifact_root + "/" + artifact_path

    with AzureMLMocks() as aml_mocks:
        result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
            mlflow.azureml.cli.commands,
            [
                "build-image",
                "-m",
                model_uri,
                "-w",
                "test_workspace",
                "-i",
                "image_name",
                "-n",
                "model_name",
            ],
        )
        assert result.exit_code == 0

        assert aml_mocks["register_model"].call_count == 1
        assert aml_mocks["create_image"].call_count == 1
        assert aml_mocks["load_workspace"].call_count == 1


@pytest.mark.large
@mock.patch("mlflow.azureml.mlflow_version", "0.7.0")
def test_cli_build_image_parses_and_includes_user_specified_tags_in_azureml_image_and_model_tags(
    sklearn_model, model_path
):
    custom_tags = {
        "User": "Corey",
        "Date": "Today",
        "Other": "Entry",
    }

    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)

    with AzureMLMocks() as aml_mocks:
        result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
            mlflow.azureml.cli.commands,
            [
                "build-image",
                "-m",
                model_path,
                "-w",
                "test_workspace",
                "-t",
                json.dumps(custom_tags),
            ],
        )
        assert result.exit_code == 0

        register_model_call_args = aml_mocks["register_model"].call_args_list
        assert len(register_model_call_args) == 1
        _, register_model_call_kwargs = register_model_call_args[0]
        called_tags = register_model_call_kwargs["tags"]
        assert custom_tags.items() <= called_tags.items()

        create_image_call_args = aml_mocks["create_image"].call_args_list
        assert len(create_image_call_args) == 1
        _, create_image_call_kwargs = create_image_call_args[0]
        image_config = create_image_call_kwargs["image_config"]
        assert custom_tags.items() <= image_config.tags.items()
