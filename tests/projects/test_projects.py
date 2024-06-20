import json
import os
import shutil
import subprocess
import uuid
from unittest import mock

import git
import pytest
import yaml

import mlflow
from mlflow import MlflowClient
from mlflow.entities import RunStatus, SourceType, ViewType
from mlflow.environment_variables import MLFLOW_CONDA_CREATE_ENV_CMD, MLFLOW_CONDA_HOME
from mlflow.exceptions import ExecutionException, MlflowException
from mlflow.projects import _parse_kubernetes_config, _resolve_experiment_id
from mlflow.store.tracking.file_store import FileStore
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.conda import CONDA_EXE, get_or_create_conda_env
from mlflow.utils.mlflow_tags import (
    LEGACY_MLFLOW_GIT_BRANCH_NAME,
    LEGACY_MLFLOW_GIT_REPO_URL,
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_REPO_URL,
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_PROJECT_BACKEND,
    MLFLOW_PROJECT_ENTRY_POINT,
    MLFLOW_PROJECT_ENV,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_USER,
)
from mlflow.utils.process import ShellCommandException

from tests.projects.utils import TEST_PROJECT_DIR, TEST_PROJECT_NAME, validate_exit_status

MOCK_USER = "janebloggs"


@pytest.fixture
def patch_user():
    with mock.patch("mlflow.projects.utils._get_user", return_value=MOCK_USER):
        yield


def _get_version_local_git_repo(local_git_repo):
    repo = git.Repo(local_git_repo, search_parent_directories=True)
    return repo.git.rev_parse("HEAD")


@pytest.fixture(scope="module", autouse=True)
def clean_mlruns_dir():
    yield
    dir_path = os.path.join(TEST_PROJECT_DIR, "mlruns")
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


@pytest.mark.parametrize(
    ("experiment_name", "experiment_id", "expected"),
    [
        ("Default", None, "0"),
        ("add an experiment", None, "1"),
        (None, 2, "2"),
        (None, "2", "2"),
        (None, None, "0"),
    ],
)
def test_resolve_experiment_id(experiment_name, experiment_id, expected):
    assert expected == _resolve_experiment_id(
        experiment_name=experiment_name, experiment_id=experiment_id
    )


def test_resolve_experiment_id_should_not_allow_both_name_and_id_in_use():
    with pytest.raises(
        MlflowException, match="Specify only one of 'experiment_name' or 'experiment_id'."
    ):
        _resolve_experiment_id(experiment_name="experiment_named", experiment_id="44")


def test_invalid_run_mode():
    """Verify that we raise an exception given an invalid run mode"""
    with pytest.raises(
        ExecutionException, match="Got unsupported execution mode some unsupported mode"
    ):
        mlflow.projects.run(uri=TEST_PROJECT_DIR, backend="some unsupported mode")


def test_expected_tags_logged_when_using_conda():
    with mock.patch.object(MlflowClient, "set_tag") as tag_mock:
        try:
            mlflow.projects.run(TEST_PROJECT_DIR, env_manager="conda")
        finally:
            tag_mock.assert_has_calls(
                [
                    mock.call(mock.ANY, MLFLOW_PROJECT_BACKEND, "local"),
                    mock.call(mock.ANY, MLFLOW_PROJECT_ENV, "conda"),
                ],
                any_order=True,
            )


@pytest.mark.usefixtures("patch_user")
@pytest.mark.parametrize("use_start_run", map(str, [0, 1]))
@pytest.mark.parametrize("version", [None, "master", "git-commit"])
def test_run_local_git_repo(
    local_git_repo, local_git_repo_uri, use_start_run, version, monkeypatch
):
    monkeypatch.setenvs(
        {
            "DATABRICKS_HOST": "my-host",
            "DATABRICKS_TOKEN": "my-token",
        }
    )
    if version is not None:
        uri = local_git_repo_uri + "#" + TEST_PROJECT_NAME
    else:
        uri = os.path.join(f"{local_git_repo}/", TEST_PROJECT_NAME)
    if version == "git-commit":
        version = _get_version_local_git_repo(local_git_repo)
    submitted_run = mlflow.projects.run(
        uri,
        entry_point="test_tracking",
        version=version,
        parameters={"use_start_run": use_start_run},
        env_manager="local",
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
    )

    # Blocking runs should be finished when they return
    validate_exit_status(submitted_run.get_status(), RunStatus.FINISHED)
    # Test that we can call wait() on a synchronous run & that the run has the correct
    # status after calling wait().
    submitted_run.wait()
    validate_exit_status(submitted_run.get_status(), RunStatus.FINISHED)
    # Validate run contents in the FileStore
    run_id = submitted_run.run_id
    mlflow_service = MlflowClient()
    runs = mlflow_service.search_runs(
        [FileStore.DEFAULT_EXPERIMENT_ID], run_view_type=ViewType.ACTIVE_ONLY
    )
    assert len(runs) == 1
    store_run_id = runs[0].info.run_id
    assert run_id == store_run_id
    run = mlflow_service.get_run(run_id)

    assert run.info.status == RunStatus.to_string(RunStatus.FINISHED)

    assert run.data.params == {
        "use_start_run": use_start_run,
    }
    assert run.data.metrics == {"some_key": 3}

    tags = run.data.tags
    assert tags[MLFLOW_USER] == MOCK_USER
    assert "file:" in tags[MLFLOW_SOURCE_NAME]
    assert tags[MLFLOW_SOURCE_TYPE] == SourceType.to_string(SourceType.PROJECT)
    assert tags[MLFLOW_PROJECT_ENTRY_POINT] == "test_tracking"
    assert tags[MLFLOW_PROJECT_BACKEND] == "local"

    if version == "master":
        assert tags[MLFLOW_GIT_BRANCH] == "master"
        assert tags[MLFLOW_GIT_REPO_URL] == local_git_repo_uri
        assert tags[LEGACY_MLFLOW_GIT_BRANCH_NAME] == "master"
        assert tags[LEGACY_MLFLOW_GIT_REPO_URL] == local_git_repo_uri


def test_invalid_version_local_git_repo(local_git_repo_uri):
    # Run project with invalid commit hash
    with pytest.raises(ExecutionException, match=r"Unable to checkout version \'badc0de\'"):
        mlflow.projects.run(
            local_git_repo_uri + "#" + TEST_PROJECT_NAME,
            entry_point="test_tracking",
            version="badc0de",
            env_manager="local",
            experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        )


@pytest.mark.parametrize("use_start_run", map(str, [0, 1]))
@pytest.mark.usefixtures("patch_user")
def test_run(use_start_run):
    submitted_run = mlflow.projects.run(
        TEST_PROJECT_DIR,
        entry_point="test_tracking",
        parameters={"use_start_run": use_start_run},
        env_manager="local",
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
    )
    assert submitted_run.run_id is not None
    # Blocking runs should be finished when they return
    validate_exit_status(submitted_run.get_status(), RunStatus.FINISHED)
    # Test that we can call wait() on a synchronous run & that the run has the correct
    # status after calling wait().
    submitted_run.wait()
    validate_exit_status(submitted_run.get_status(), RunStatus.FINISHED)
    # Validate run contents in the FileStore
    run_id = submitted_run.run_id
    mlflow_service = MlflowClient()

    runs = mlflow_service.search_runs(
        [FileStore.DEFAULT_EXPERIMENT_ID], run_view_type=ViewType.ACTIVE_ONLY
    )
    assert len(runs) == 1
    store_run_id = runs[0].info.run_id
    assert run_id == store_run_id
    run = mlflow_service.get_run(run_id)

    assert run.info.status == RunStatus.to_string(RunStatus.FINISHED)

    assert run.data.params == {
        "use_start_run": use_start_run,
    }
    assert run.data.metrics == {"some_key": 3}

    tags = run.data.tags
    assert tags[MLFLOW_USER] == MOCK_USER
    assert "file:" in tags[MLFLOW_SOURCE_NAME]
    assert tags[MLFLOW_SOURCE_TYPE] == SourceType.to_string(SourceType.PROJECT)
    assert tags[MLFLOW_PROJECT_ENTRY_POINT] == "test_tracking"


def test_run_with_parent():
    """Verify that if we are in a nested run, mlflow.projects.run() will have a parent_run_id."""
    with mlflow.start_run():
        parent_run_id = mlflow.active_run().info.run_id
        submitted_run = mlflow.projects.run(
            TEST_PROJECT_DIR,
            entry_point="test_tracking",
            parameters={"use_start_run": "1"},
            env_manager="local",
            experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        )
    assert submitted_run.run_id is not None
    validate_exit_status(submitted_run.get_status(), RunStatus.FINISHED)
    run_id = submitted_run.run_id
    run = MlflowClient().get_run(run_id)
    assert run.data.tags[MLFLOW_PARENT_RUN_ID] == parent_run_id


def test_run_with_artifact_path(tmp_path):
    artifact_file = tmp_path.joinpath("model.pkl")
    artifact_file.write_text("Hello world")
    with mlflow.start_run() as run:
        mlflow.log_artifact(artifact_file)
        submitted_run = mlflow.projects.run(
            TEST_PROJECT_DIR,
            entry_point="test_artifact_path",
            parameters={"model": f"runs:/{run.info.run_id}/model.pkl"},
            env_manager="local",
            experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        )
        validate_exit_status(submitted_run.get_status(), RunStatus.FINISHED)


def test_run_async():
    submitted_run0 = mlflow.projects.run(
        TEST_PROJECT_DIR,
        entry_point="sleep",
        parameters={"duration": 2},
        env_manager="local",
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        synchronous=False,
    )
    validate_exit_status(submitted_run0.get_status(), RunStatus.RUNNING)
    submitted_run0.wait()
    validate_exit_status(submitted_run0.get_status(), RunStatus.FINISHED)
    submitted_run1 = mlflow.projects.run(
        TEST_PROJECT_DIR,
        entry_point="sleep",
        parameters={"duration": -1, "invalid-param": 30},
        env_manager="local",
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        synchronous=False,
    )
    submitted_run1.wait()
    validate_exit_status(submitted_run1.get_status(), RunStatus.FAILED)


@pytest.mark.parametrize(
    ("mock_env", "expected_conda", "expected_activate"),
    [
        (
            {CONDA_EXE: "/abc/conda"},
            "/abc/conda",
            "/abc/activate",
        ),
        (
            {MLFLOW_CONDA_HOME.name: "/some/dir/"},
            "/some/dir/bin/conda",
            "/some/dir/bin/activate",
        ),
    ],
)
def test_conda_path(mock_env, expected_conda, expected_activate, monkeypatch):
    """Verify that we correctly determine the path to conda executables"""
    monkeypatch.delenvs([CONDA_EXE, MLFLOW_CONDA_HOME.name], raising=False)
    monkeypatch.setenvs(mock_env)
    assert mlflow.utils.conda.get_conda_bin_executable("conda") == expected_conda
    assert mlflow.utils.conda.get_conda_bin_executable("activate") == expected_activate


@pytest.mark.parametrize(
    ("mock_env", "expected_conda_env_create_path"),
    [
        (
            {CONDA_EXE: "/abc/conda"},
            "/abc/conda",
        ),
        (
            {CONDA_EXE: "/abc/conda", MLFLOW_CONDA_CREATE_ENV_CMD.name: "mamba"},
            "/abc/mamba",
        ),
        (
            {MLFLOW_CONDA_HOME.name: "/some/dir/"},
            "/some/dir/bin/conda",
        ),
        (
            {MLFLOW_CONDA_HOME.name: "/some/dir/", MLFLOW_CONDA_CREATE_ENV_CMD.name: "mamba"},
            "/some/dir/bin/mamba",
        ),
    ],
)
def test_find_conda_executables(mock_env, expected_conda_env_create_path, monkeypatch):
    """
    Verify that we correctly determine the path to executables to be used to
    create environments (for example, it could be mamba instead of conda)
    """
    monkeypatch.delenvs(
        [CONDA_EXE, MLFLOW_CONDA_HOME.name, MLFLOW_CONDA_CREATE_ENV_CMD.name], raising=False
    )
    monkeypatch.setenvs(mock_env)
    conda_env_create_path = mlflow.utils.conda._get_conda_executable_for_create_env()
    assert conda_env_create_path == expected_conda_env_create_path


def test_create_env_with_mamba(monkeypatch):
    """
    Test that mamba is called when set, and that we fail when mamba is not available or is
    not working. We mock the calls so we do not actually execute mamba (which is not
    installed in the test environment anyway)
    """

    def exec_cmd_mock(cmd, *args, **kwargs):
        if cmd[-1] == "--json":
            # We are supposed to list environments in JSON format
            return subprocess.CompletedProcess(
                cmd, 0, json.dumps({"envs": ["mlflow-mock-environment"]}), None
            )
        else:
            # Here we are creating the environment, no need to return
            # anything
            return subprocess.CompletedProcess(cmd, 0)

    def exec_cmd_mock_raise(cmd, *args, **kwargs):
        if os.path.basename(cmd[0]) == "mamba":
            raise OSError()

    conda_env_path = os.path.join(TEST_PROJECT_DIR, "conda.yaml")

    monkeypatch.setenv(MLFLOW_CONDA_CREATE_ENV_CMD.name, "mamba")
    # Simulate success
    with mock.patch("mlflow.utils.process._exec_cmd", side_effect=exec_cmd_mock):
        mlflow.utils.conda.get_or_create_conda_env(conda_env_path)

    # Simulate a non-working or non-existent mamba
    with mock.patch("mlflow.utils.process._exec_cmd", side_effect=exec_cmd_mock_raise):
        with pytest.raises(
            ExecutionException,
            match="You have set the env variable MLFLOW_CONDA_CREATE_ENV_CMD",
        ):
            mlflow.utils.conda.get_or_create_conda_env(conda_env_path)


def test_conda_environment_cleaned_up_when_pip_fails(tmp_path):
    conda_yaml = tmp_path / "conda.yaml"
    content = f"""
name: {uuid.uuid4().hex}
channels:
  - conda-forge
dependencies:
  - python={PYTHON_VERSION}
  - pip
  - pip:
      - mlflow==999.999.999
"""
    conda_yaml.write_text(content)
    envs_before = mlflow.utils.conda._list_conda_environments()

    # `conda create` should fail because mlflow 999.999.999 doesn't exist
    with pytest.raises(ShellCommandException, match=r"No matching distribution found"):
        mlflow.utils.conda.get_or_create_conda_env(conda_yaml, capture_output=True)

    # Ensure the environment is cleaned up
    envs_after = mlflow.utils.conda._list_conda_environments()
    assert envs_before == envs_after


def test_cancel_run():
    submitted_run0, submitted_run1 = (
        mlflow.projects.run(
            TEST_PROJECT_DIR,
            entry_point="sleep",
            parameters={"duration": 2},
            env_manager="local",
            experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
            synchronous=False,
        )
        for _ in range(2)
    )
    submitted_run0.cancel()
    validate_exit_status(submitted_run0.get_status(), RunStatus.FAILED)
    # Sanity check: cancelling one run has no effect on the other
    assert submitted_run1.wait()
    validate_exit_status(submitted_run1.get_status(), RunStatus.FINISHED)
    # Try cancelling after calling wait()
    submitted_run1.cancel()
    validate_exit_status(submitted_run1.get_status(), RunStatus.FINISHED)


def test_parse_kubernetes_config():
    work_dir = "./examples/docker"
    kubernetes_config = {
        "kube-context": "docker-for-desktop",
        "kube-job-template-path": os.path.join(work_dir, "kubernetes_job_template.yaml"),
        "repository-uri": "dockerhub_account/mlflow-kubernetes-example",
    }
    yaml_obj = None
    with open(kubernetes_config["kube-job-template-path"]) as job_template:
        yaml_obj = yaml.safe_load(job_template.read())
    kube_config = _parse_kubernetes_config(kubernetes_config)
    assert kube_config["kube-context"] == kubernetes_config["kube-context"]
    assert kube_config["kube-job-template-path"] == kubernetes_config["kube-job-template-path"]
    assert kube_config["repository-uri"] == kubernetes_config["repository-uri"]
    assert kube_config["kube-job-template"] == yaml_obj


@pytest.fixture
def mock_kubernetes_job_template(tmp_path):
    k8s_yaml = tmp_path.joinpath("kubernetes_job_template.yaml")
    k8s_yaml.write_text(
        """
apiVersion: batch/v1
kind: Job
metadata:
  name: "{replaced with MLflow Project name}"
  namespace: mlflow
spec:
  ttlSecondsAfterFinished: 100
  backoffLimit: 0
  template:
    spec:
      containers:
      - name: "{replaced with MLflow Project name}"
        image: "{replaced with URI of Docker image created during Project execution}"
        command: ["{replaced with MLflow Project entry point command}"]
        resources:
          limits:
            memory: 512Mi
          requests:
            memory: 256Mi
      restartPolicy: Never
""".lstrip()
    )
    return str(k8s_yaml)


class StartsWithMatcher:
    def __init__(self, prefix):
        self.prefix = prefix

    def __eq__(self, other):
        return isinstance(other, str) and other.startswith(self.prefix)


def test_parse_kubernetes_config_without_context(mock_kubernetes_job_template):
    with mock.patch("mlflow.projects._logger.debug") as mock_debug:
        kubernetes_config = {
            "repository-uri": "dockerhub_account/mlflow-kubernetes-example",
            "kube-job-template-path": mock_kubernetes_job_template,
        }
        _parse_kubernetes_config(kubernetes_config)
        mock_debug.assert_called_once_with(
            StartsWithMatcher("Could not find kube-context in backend_config")
        )


def test_parse_kubernetes_config_without_image_uri(mock_kubernetes_job_template):
    kubernetes_config = {
        "kube-context": "docker-for-desktop",
        "kube-job-template-path": mock_kubernetes_job_template,
    }
    with pytest.raises(ExecutionException, match="Could not find 'repository-uri'"):
        _parse_kubernetes_config(kubernetes_config)


def test_parse_kubernetes_config_invalid_template_job_file():
    kubernetes_config = {
        "kube-context": "docker-for-desktop",
        "repository-uri": "username/mlflow-kubernetes-example",
        "kube-job-template-path": "file_not_found.yaml",
    }
    with pytest.raises(ExecutionException, match="Could not find 'kube-job-template-path'"):
        _parse_kubernetes_config(kubernetes_config)


@pytest.mark.parametrize("synchronous", [True, False])
def test_credential_propagation(synchronous, monkeypatch):
    class DummyProcess:
        def wait(self):
            return 0

        def poll(self):
            return 0

        def communicate(self, _):
            return "", ""

    monkeypatch.setenvs(
        {
            "DATABRICKS_HOST": "host",
            "DATABRICKS_TOKEN": "mytoken",
        }
    )
    with mock.patch("subprocess.Popen", return_value=DummyProcess()) as popen_mock, mock.patch(
        "mlflow.utils.uri.is_databricks_uri", return_value=True
    ):
        mlflow.projects.run(
            TEST_PROJECT_DIR,
            entry_point="sleep",
            experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
            parameters={"duration": 2},
            env_manager="local",
            synchronous=synchronous,
        )
        _, kwargs = popen_mock.call_args
        env = kwargs["env"]
        assert env["DATABRICKS_HOST"] == "host"
        assert env["DATABRICKS_TOKEN"] == "mytoken"


def test_get_or_create_conda_env_capture_output_mode(tmp_path):
    conda_yaml_file = tmp_path / "conda.yaml"
    conda_yaml_file.write_text(
        """
channels:
- conda-forge
dependencies:
- pip:
  - scikit-learn==99.99.99
"""
    )
    with pytest.raises(
        ShellCommandException,
        match="Could not find a version that satisfies the requirement scikit-learn==99.99.99",
    ):
        get_or_create_conda_env(str(conda_yaml_file), capture_output=True)
