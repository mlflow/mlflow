import os

import pytest
from unittest import mock

from databricks_cli.configure.provider import DatabricksConfig

import mlflow
from mlflow.entities import ViewType
from mlflow.projects.docker import _get_docker_image_uri
from mlflow.projects import ExecutionException
from mlflow.projects.backend.local import _get_docker_command
from mlflow.store.tracking import file_store
from mlflow.utils.mlflow_tags import (
    MLFLOW_PROJECT_ENV,
    MLFLOW_PROJECT_BACKEND,
    MLFLOW_DOCKER_IMAGE_URI,
    MLFLOW_DOCKER_IMAGE_ID,
)
from tests.projects.utils import TEST_DOCKER_PROJECT_DIR
from tests.projects.utils import docker_example_base_image  # pylint: disable=unused-import
from mlflow.projects import _project_spec
from mlflow.exceptions import MlflowException


def _build_uri(base_uri, subdirectory):
    if subdirectory != "":
        return "%s#%s" % (base_uri, subdirectory)
    return base_uri


@pytest.mark.parametrize("use_start_run", map(str, [0, 1]))
@pytest.mark.large
def test_docker_project_execution(
    use_start_run, tmpdir, docker_example_base_image
):  # pylint: disable=unused-argument
    expected_params = {"use_start_run": use_start_run}
    submitted_run = mlflow.projects.run(
        TEST_DOCKER_PROJECT_DIR,
        experiment_id=file_store.FileStore.DEFAULT_EXPERIMENT_ID,
        parameters=expected_params,
        entry_point="test_tracking",
        docker_args={"memory": "1g", "privileged": True},
    )
    # Validate run contents in the FileStore
    run_id = submitted_run.run_id
    mlflow_service = mlflow.tracking.MlflowClient()
    run_infos = mlflow_service.list_run_infos(
        experiment_id=file_store.FileStore.DEFAULT_EXPERIMENT_ID, run_view_type=ViewType.ACTIVE_ONLY
    )
    assert len(run_infos) == 1
    store_run_id = run_infos[0].run_id
    assert run_id == store_run_id
    run = mlflow_service.get_run(run_id)
    assert run.data.params == expected_params
    assert run.data.metrics == {"some_key": 3}
    exact_expected_tags = {
        MLFLOW_PROJECT_ENV: "docker",
        MLFLOW_PROJECT_BACKEND: "local",
    }
    approx_expected_tags = {
        MLFLOW_DOCKER_IMAGE_URI: "docker-example",
        MLFLOW_DOCKER_IMAGE_ID: "sha256:",
    }
    run_tags = run.data.tags
    for k, v in exact_expected_tags.items():
        assert run_tags[k] == v
    for k, v in approx_expected_tags.items():
        assert run_tags[k].startswith(v)
    artifacts = mlflow_service.list_artifacts(run_id=run_id)
    assert len(artifacts) == 1
    docker_cmd = submitted_run.command_proc.args[2]
    assert "--memory 1g" in docker_cmd
    assert "--privileged" in docker_cmd


@pytest.mark.large
def test_docker_project_execution_async_docker_args(
    tmpdir, docker_example_base_image
):  # pylint: disable=unused-argument
    submitted_run = mlflow.projects.run(
        TEST_DOCKER_PROJECT_DIR,
        experiment_id=file_store.FileStore.DEFAULT_EXPERIMENT_ID,
        parameters={"use_start_run": "0"},
        entry_point="test_tracking",
        docker_args={"memory": "1g", "privileged": True},
        synchronous=False,
    )
    submitted_run.wait()

    args = submitted_run.command_proc.args
    assert len([a for a in args if a == "--docker-args"]) == 2
    first_idx = args.index("--docker-args")
    second_idx = args.index("--docker-args", first_idx + 1)
    assert args[first_idx + 1] == "memory=1g"
    assert args[second_idx + 1] == "privileged"


@pytest.mark.parametrize(
    "tracking_uri, expected_command_segment",
    [
        (None, "-e MLFLOW_TRACKING_URI=/mlflow/tmp/mlruns"),
        ("http://some-tracking-uri", "-e MLFLOW_TRACKING_URI=http://some-tracking-uri"),
        ("databricks://some-profile", "-e MLFLOW_TRACKING_URI=databricks "),
    ],
)
@mock.patch("databricks_cli.configure.provider.ProfileConfigProvider")
@pytest.mark.large
def test_docker_project_tracking_uri_propagation(
    ProfileConfigProvider, tmpdir, tracking_uri, expected_command_segment, docker_example_base_image
):  # pylint: disable=unused-argument
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = DatabricksConfig.from_password(
        "host", "user", "pass", insecure=True
    )
    ProfileConfigProvider.return_value = mock_provider
    # Create and mock local tracking directory
    local_tracking_dir = os.path.join(tmpdir.strpath, "mlruns")
    if tracking_uri is None:
        tracking_uri = local_tracking_dir
    old_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        with mock.patch("mlflow.tracking._tracking_service.utils._get_store") as _get_store_mock:
            _get_store_mock.return_value = file_store.FileStore(local_tracking_dir)
            mlflow.projects.run(
                TEST_DOCKER_PROJECT_DIR, experiment_id=file_store.FileStore.DEFAULT_EXPERIMENT_ID
            )
    finally:
        mlflow.set_tracking_uri(old_uri)


def test_docker_uri_mode_validation(docker_example_base_image):  # pylint: disable=unused-argument
    with pytest.raises(ExecutionException, match="When running on Databricks"):
        mlflow.projects.run(TEST_DOCKER_PROJECT_DIR, backend="databricks", backend_config={})


@mock.patch("mlflow.projects.docker._get_git_commit")
def test_docker_image_uri_with_git(get_git_commit_mock):
    get_git_commit_mock.return_value = "1234567890"
    image_uri = _get_docker_image_uri("my_project", "my_workdir")
    assert image_uri == "my_project:1234567"
    get_git_commit_mock.assert_called_with("my_workdir")


@mock.patch("mlflow.projects.docker._get_git_commit")
def test_docker_image_uri_no_git(get_git_commit_mock):
    get_git_commit_mock.return_value = None
    image_uri = _get_docker_image_uri("my_project", "my_workdir")
    assert image_uri == "my_project"
    get_git_commit_mock.assert_called_with("my_workdir")


def test_docker_valid_project_backend_local():
    work_dir = "./examples/docker"
    project = _project_spec.load_project(work_dir)
    mlflow.projects.docker.validate_docker_env(project)


def test_docker_invalid_project_backend_local():
    work_dir = "./examples/docker"
    project = _project_spec.load_project(work_dir)
    project.name = None
    with pytest.raises(ExecutionException, match="Project name in MLProject must be specified"):
        mlflow.projects.docker.validate_docker_env(project)


@pytest.mark.parametrize(
    "artifact_uri, host_artifact_uri, container_artifact_uri, should_mount",
    [
        ("/tmp/mlruns/artifacts", "/tmp/mlruns/artifacts", "/tmp/mlruns/artifacts", True),
        ("s3://my_bucket", None, None, False),
        ("file:///tmp/mlruns/artifacts", "/tmp/mlruns/artifacts", "/tmp/mlruns/artifacts", True),
        ("./mlruns", os.path.abspath("./mlruns"), "/mlflow/projects/code/mlruns", True),
    ],
)
def test_docker_mount_local_artifact_uri(
    artifact_uri, host_artifact_uri, container_artifact_uri, should_mount
):
    active_run = mock.MagicMock()
    run_info = mock.MagicMock()
    run_info.run_id = "fake_run_id"
    run_info.experiment_id = "fake_experiment_id"
    run_info.artifact_uri = artifact_uri
    active_run.info = run_info
    image = mock.MagicMock()
    image.tags = ["image:tag"]

    docker_command = _get_docker_command(image, active_run)

    docker_volume_expected = "-v {}:{}".format(host_artifact_uri, container_artifact_uri)
    assert (docker_volume_expected in " ".join(docker_command)) == should_mount


@mock.patch("databricks_cli.configure.provider.ProfileConfigProvider")
def test_docker_databricks_tracking_cmd_and_envs(ProfileConfigProvider):
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = DatabricksConfig.from_password(
        "host", "user", "pass", insecure=True
    )
    ProfileConfigProvider.return_value = mock_provider

    cmds, envs = mlflow.projects.docker.get_docker_tracking_cmd_and_envs(
        "databricks://some-profile"
    )
    assert envs == {
        "DATABRICKS_HOST": "host",
        "DATABRICKS_USERNAME": "user",
        "DATABRICKS_PASSWORD": "pass",
        "DATABRICKS_INSECURE": "True",
        mlflow.tracking._TRACKING_URI_ENV_VAR: "databricks",
    }
    assert cmds == []


@pytest.mark.parametrize(
    "volumes, environment, os_environ, expected",
    [
        ([], ["VAR1"], {"VAR1": "value1"}, [("-e", "VAR1=value1")]),
        ([], ["VAR1"], {}, ["should_crash", ("-e", "VAR1=value1")]),
        ([], ["VAR1"], {"OTHER_VAR": "value1"}, ["should_crash", ("-e", "VAR1=value1")]),
        (
            [],
            ["VAR1", ["VAR2", "value2"]],
            {"VAR1": "value1"},
            [("-e", "VAR1=value1"), ("-e", "VAR2=value2")],
        ),
        ([], [["VAR2", "value2"]], {"VAR1": "value1"}, [("-e", "VAR2=value2")]),
        (
            ["/path:/path"],
            ["VAR1"],
            {"VAR1": "value1"},
            [("-e", "VAR1=value1"), ("-v", "/path:/path")],
        ),
        (
            ["/path:/path"],
            [["VAR2", "value2"]],
            {"VAR1": "value1"},
            [("-e", "VAR2=value2"), ("-v", "/path:/path")],
        ),
    ],
)
def test_docker_user_specified_env_vars(volumes, environment, expected, os_environ):
    active_run = mock.MagicMock()
    run_info = mock.MagicMock()
    run_info.run_id = "fake_run_id"
    run_info.experiment_id = "fake_experiment_id"
    run_info.artifact_uri = "/tmp/mlruns/artifacts"
    active_run.info = run_info
    image = mock.MagicMock()
    image.tags = ["image:tag"]

    if "should_crash" in expected:
        expected.remove("should_crash")
        with pytest.raises(MlflowException, match="This project expects"):
            with mock.patch.dict("os.environ", os_environ):
                _get_docker_command(image, active_run, None, volumes, environment)
    else:
        with mock.patch.dict("os.environ", os_environ):
            docker_command = _get_docker_command(image, active_run, None, volumes, environment)
        for exp_type, expected in expected:
            assert expected in docker_command
            assert docker_command[docker_command.index(expected) - 1] == exp_type


@pytest.mark.parametrize("docker_args", [{}, {"ARG": "VAL"}, {"ARG1": "VAL1", "ARG2": "VAL2"}])
def test_docker_run_args(docker_args):
    active_run = mock.MagicMock()
    run_info = mock.MagicMock()
    run_info.run_id = "fake_run_id"
    run_info.experiment_id = "fake_experiment_id"
    run_info.artifact_uri = "/tmp/mlruns/artifacts"
    active_run.info = run_info
    image = mock.MagicMock()
    image.tags = ["image:tag"]

    docker_command = _get_docker_command(image, active_run, docker_args, None, None)

    for flag, value in docker_args.items():
        assert docker_command[docker_command.index(value) - 1] == "--{}".format(flag)
