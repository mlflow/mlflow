"""
The ``mlflow.projects`` module provides an API for running MLflow projects locally or remotely.
"""
import json
import yaml
import os
import logging

import mlflow.projects.databricks
from mlflow import tracking
from mlflow.entities import RunStatus
from mlflow.exceptions import ExecutionException, MlflowException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects.utils import (
    PROJECT_SYNCHRONOUS,
    get_entry_point_command,
    get_run_env_vars,
    fetch_and_validate_project,
    get_or_create_run,
    load_project,
    MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG,
    PROJECT_ENV_MANAGER,
    PROJECT_STORAGE_DIR,
    PROJECT_DOCKER_ARGS,
    PROJECT_BUILD_IMAGE,
    PROJECT_DOCKER_AUTH,
)
from mlflow.projects.backend import loader
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.mlflow_tags import (
    MLFLOW_PROJECT_ENV,
    MLFLOW_PROJECT_BACKEND,
    MLFLOW_RUN_NAME,
    MLFLOW_DOCKER_IMAGE_ID,
)
from mlflow.utils import env_manager as _EnvManager
import mlflow.utils.uri

_logger = logging.getLogger(__name__)


def _resolve_experiment_id(experiment_name=None, experiment_id=None):
    """
    Resolve experiment.

    Verifies either one or other is specified - cannot be both selected.

    If ``experiment_name`` is provided and does not exist, an experiment
    of that name is created and its id is returned.

    :param experiment_name: Name of experiment under which to launch the run.
    :param experiment_id: ID of experiment under which to launch the run.
    :return: str
    """

    if experiment_name and experiment_id:
        raise MlflowException("Specify only one of 'experiment_name' or 'experiment_id'.")

    if experiment_id:
        return str(experiment_id)

    if experiment_name:
        client = tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp:
            return exp.experiment_id
        else:
            _logger.info("'%s' does not exist. Creating a new experiment", experiment_name)
            return client.create_experiment(experiment_name)

    return _get_experiment_id()


def _run(
    uri,
    experiment_id,
    entry_point,
    version,
    parameters,
    docker_args,
    backend_name,
    backend_config,
    storage_dir,
    env_manager,
    synchronous,
    run_name,
    build_image,
    docker_auth,
):
    """
    Helper that delegates to the project-running method corresponding to the passed-in backend.
    Returns a ``SubmittedRun`` corresponding to the project run.
    """
    tracking_store_uri = tracking.get_tracking_uri()
    backend_config[PROJECT_ENV_MANAGER] = env_manager
    backend_config[PROJECT_SYNCHRONOUS] = synchronous
    backend_config[PROJECT_DOCKER_ARGS] = docker_args
    backend_config[PROJECT_STORAGE_DIR] = storage_dir
    backend_config[PROJECT_BUILD_IMAGE] = build_image
    backend_config[PROJECT_DOCKER_AUTH] = docker_auth
    # TODO: remove this check once kubernetes execution has been refactored
    if backend_name not in {"databricks", "kubernetes"}:
        backend = loader.load_backend(backend_name)
        if backend:
            submitted_run = backend.run(
                uri,
                entry_point,
                parameters,
                version,
                backend_config,
                tracking_store_uri,
                experiment_id,
            )
            tracking.MlflowClient().set_tag(
                submitted_run.run_id, MLFLOW_PROJECT_BACKEND, backend_name
            )
            if run_name is not None:
                tracking.MlflowClient().set_tag(submitted_run.run_id, MLFLOW_RUN_NAME, run_name)
            return submitted_run

    work_dir = fetch_and_validate_project(uri, version, entry_point, parameters)
    project = load_project(work_dir)
    _validate_execution_environment(project, backend_name)

    active_run = get_or_create_run(
        None, uri, experiment_id, work_dir, version, entry_point, parameters
    )

    if run_name is not None:
        tracking.MlflowClient().set_tag(active_run.info.run_id, MLFLOW_RUN_NAME, run_name)

    if backend_name == "databricks":
        tracking.MlflowClient().set_tag(
            active_run.info.run_id, MLFLOW_PROJECT_BACKEND, "databricks"
        )
        from mlflow.projects.databricks import run_databricks

        return run_databricks(
            remote_run=active_run,
            uri=uri,
            entry_point=entry_point,
            work_dir=work_dir,
            parameters=parameters,
            experiment_id=experiment_id,
            cluster_spec=backend_config,
            env_manager=env_manager,
        )

    elif backend_name == "kubernetes":
        from mlflow.projects.docker import (
            build_docker_image,
            validate_docker_env,
            validate_docker_installation,
        )
        from mlflow.projects import kubernetes as kb

        tracking.MlflowClient().set_tag(active_run.info.run_id, MLFLOW_PROJECT_ENV, "docker")
        tracking.MlflowClient().set_tag(
            active_run.info.run_id, MLFLOW_PROJECT_BACKEND, "kubernetes"
        )
        validate_docker_env(project)
        validate_docker_installation()
        kube_config = _parse_kubernetes_config(backend_config)
        image = build_docker_image(
            work_dir=work_dir,
            repository_uri=kube_config["repository-uri"],
            base_image=project.docker_env.get("image"),
            run_id=active_run.info.run_id,
            build_image=build_image,
            docker_auth=docker_auth,
        )
        image_digest = kb.push_image_to_registry(image.tags[0])
        tracking.MlflowClient().set_tag(
            active_run.info.run_id, MLFLOW_DOCKER_IMAGE_ID, image_digest
        )
        submitted_run = kb.run_kubernetes_job(
            project.name,
            active_run,
            image.tags[0],
            image_digest,
            get_entry_point_command(project, entry_point, parameters, storage_dir),
            get_run_env_vars(
                run_id=active_run.info.run_uuid, experiment_id=active_run.info.experiment_id
            ),
            kube_config.get("kube-context", None),
            kube_config["kube-job-template"],
        )
        return submitted_run

    supported_backends = ["databricks", "kubernetes"] + list(loader.MLFLOW_BACKENDS.keys())
    raise ExecutionException(
        "Got unsupported execution mode %s. Supported "
        "values: %s" % (backend_name, supported_backends)
    )


def run(
    uri,
    entry_point="main",
    version=None,
    parameters=None,
    docker_args=None,
    experiment_name=None,
    experiment_id=None,
    backend="local",
    backend_config=None,
    storage_dir=None,
    synchronous=True,
    run_id=None,
    run_name=None,
    env_manager=None,
    build_image=False,
    docker_auth=None,
):
    """
    Run an MLflow project. The project can be local or stored at a Git URI.

    MLflow provides built-in support for running projects locally or remotely on a Databricks or
    Kubernetes cluster. You can also run projects against other targets by installing an appropriate
    third-party plugin. See `Community Plugins <../plugins.html#community-plugins>`_ for more
    information.

    For information on using this method in chained workflows, see `Building Multistep Workflows
    <../projects.html#building-multistep-workflows>`_.

    :raises: :py:class:`mlflow.exceptions.ExecutionException` If a run launched in blocking mode
             is unsuccessful.

    :param uri: URI of project to run. A local filesystem path
                or a Git repository URI (e.g. https://github.com/mlflow/mlflow-example)
                pointing to a project directory containing an MLproject file.
    :param entry_point: Entry point to run within the project. If no entry point with the specified
                        name is found, runs the project file ``entry_point`` as a script,
                        using "python" to run ``.py`` files and the default shell (specified by
                        environment variable ``$SHELL``) to run ``.sh`` files.
    :param version: For Git-based projects, either a commit hash or a branch name.
    :param parameters: Parameters (dictionary) for the entry point command.
    :param docker_args: Arguments (dictionary) for the docker command.
    :param experiment_name: Name of experiment under which to launch the run.
    :param experiment_id: ID of experiment under which to launch the run.
    :param backend: Execution backend for the run: MLflow provides built-in support for "local",
                    "databricks", and "kubernetes" (experimental) backends. If running against
                    Databricks, will run against a Databricks workspace determined as follows:
                    if a Databricks tracking URI of the form ``databricks://profile`` has been set
                    (e.g. by setting the MLFLOW_TRACKING_URI environment variable), will run
                    against the workspace specified by <profile>. Otherwise, runs against the
                    workspace specified by the default Databricks CLI profile.
    :param backend_config: A dictionary, or a path to a JSON file (must end in '.json'), which will
                           be passed as config to the backend. The exact content which should be
                           provided is different for each execution backend and is documented
                           at https://www.mlflow.org/docs/latest/projects.html.
    :param storage_dir: Used only if ``backend`` is "local". MLflow downloads artifacts from
                        distributed URIs passed to parameters of type ``path`` to subdirectories of
                        ``storage_dir``.
    :param synchronous: Whether to block while waiting for a run to complete. Defaults to True.
                        Note that if ``synchronous`` is False and ``backend`` is "local", this
                        method will return, but the current process will block when exiting until
                        the local run completes. If the current process is interrupted, any
                        asynchronous runs launched via this method will be terminated. If
                        ``synchronous`` is True and the run fails, the current process will
                        error out as well.
    :param run_id: Note: this argument is used internally by the MLflow project APIs and should
                   not be specified. If specified, the run ID will be used instead of
                   creating a new run.
    :param run_name: The name to give the MLflow Run associated with the project execution.
                     If ``None``, the MLflow Run name is left unset.
    :param env_manager: Specify an environment manager to create a new environment for the run and
                        install project dependencies within that environment. The following values
                        are supported:

                        - local: use the local environment
                        - virtualenv: use virtualenv (and pyenv for Python version management)
                        - conda: use conda

                        If unspecified, MLflow automatically determines the environment manager to
                        use by inspecting files in the project directory. For example, if
                        ``python_env.yaml`` is present, virtualenv will be used.
    :param build_image: Whether to build a new docker image of the project or to reuse an existing
                        image. Default: False (reuse an existing image)
    :param docker_auth: A dictionary representing information to authenticate with a Docker
                        registry. See `docker.client.DockerClient.login
                        <https://docker-py.readthedocs.io/en/stable/client.html#docker.client.DockerClient.login>`_
                        for available options.
    :return: :py:class:`mlflow.projects.SubmittedRun` exposing information (e.g. run ID)
             about the launched run.

    .. code-block:: python
        :caption: Example

        import mlflow

        project_uri = "https://github.com/mlflow/mlflow-example"
        params = {"alpha": 0.5, "l1_ratio": 0.01}

        # Run MLflow project and create a reproducible conda environment
        # on a local host
        mlflow.run(project_uri, parameters=params)

    .. code-block:: text
        :caption: Output

        ...
        ...
        Elasticnet model (alpha=0.500000, l1_ratio=0.010000):
        RMSE: 0.788347345611717
        MAE: 0.6155576449938276
        R2: 0.19729662005412607
        ... mlflow.projects: === Run (ID '6a5109febe5e4a549461e149590d0a7c') succeeded ===
    """
    backend_config_dict = backend_config if backend_config is not None else {}
    if (
        backend_config
        and type(backend_config) != dict
        and os.path.splitext(backend_config)[-1] == ".json"
    ):
        with open(backend_config) as handle:
            try:
                backend_config_dict = json.load(handle)
            except ValueError:
                _logger.error(
                    "Error when attempting to load and parse JSON cluster spec from file %s",
                    backend_config,
                )
                raise

    if env_manager is not None:
        _EnvManager.validate(env_manager)

    if backend == "databricks":
        mlflow.projects.databricks.before_run_validations(mlflow.get_tracking_uri(), backend_config)
    elif backend == "local" and run_id is not None:
        backend_config_dict[MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG] = run_id

    experiment_id = _resolve_experiment_id(
        experiment_name=experiment_name, experiment_id=experiment_id
    )

    submitted_run_obj = _run(
        uri=uri,
        experiment_id=experiment_id,
        entry_point=entry_point,
        version=version,
        parameters=parameters,
        docker_args=docker_args,
        backend_name=backend,
        backend_config=backend_config_dict,
        env_manager=env_manager,
        storage_dir=storage_dir,
        synchronous=synchronous,
        run_name=run_name,
        build_image=build_image,
        docker_auth=docker_auth,
    )
    if synchronous:
        _wait_for(submitted_run_obj)
    return submitted_run_obj


def _wait_for(submitted_run_obj):
    """Wait on the passed-in submitted run, reporting its status to the tracking server."""
    run_id = submitted_run_obj.run_id
    active_run = None
    # Note: there's a small chance we fail to report the run's status to the tracking server if
    # we're interrupted before we reach the try block below
    try:
        active_run = tracking.MlflowClient().get_run(run_id) if run_id is not None else None
        if submitted_run_obj.wait():
            _logger.info("=== Run (ID '%s') succeeded ===", run_id)
            _maybe_set_run_terminated(active_run, "FINISHED")
        else:
            _maybe_set_run_terminated(active_run, "FAILED")
            raise ExecutionException("Run (ID '%s') failed" % run_id)
    except KeyboardInterrupt:
        _logger.error("=== Run (ID '%s') interrupted, cancelling run ===", run_id)
        submitted_run_obj.cancel()
        _maybe_set_run_terminated(active_run, "FAILED")
        raise


def _maybe_set_run_terminated(active_run, status):
    """
    If the passed-in active run is defined and still running (i.e. hasn't already been terminated
    within user code), mark it as terminated with the passed-in status.
    """
    if active_run is None:
        return
    run_id = active_run.info.run_id
    cur_status = tracking.MlflowClient().get_run(run_id).info.status
    if RunStatus.is_terminated(cur_status):
        return
    tracking.MlflowClient().set_terminated(run_id, status)


def _validate_execution_environment(project, backend):
    if project.docker_env and backend == "databricks":
        raise ExecutionException(
            "Running docker-based projects on Databricks is not yet supported."
        )


def _parse_kubernetes_config(backend_config):
    """
    Creates build context tarfile containing Dockerfile and project code, returning path to tarfile
    """
    if not backend_config:
        raise ExecutionException("Backend_config file not found.")
    kube_config = backend_config.copy()
    if "kube-job-template-path" not in backend_config.keys():
        raise ExecutionException(
            "'kube-job-template-path' attribute must be specified in backend_config."
        )
    kube_job_template = backend_config["kube-job-template-path"]
    if os.path.exists(kube_job_template):
        with open(kube_job_template) as job_template:
            yaml_obj = yaml.safe_load(job_template.read())
        kube_job_template = yaml_obj
        kube_config["kube-job-template"] = kube_job_template
    else:
        raise ExecutionException(f"Could not find 'kube-job-template-path': {kube_job_template}")
    if "kube-context" not in backend_config.keys():
        _logger.debug(
            "Could not find kube-context in backend_config."
            " Using current context or in-cluster config."
        )
    if "repository-uri" not in backend_config.keys():
        raise ExecutionException("Could not find 'repository-uri' in backend_config.")
    return kube_config


__all__ = ["run", "SubmittedRun"]
