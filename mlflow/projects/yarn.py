import time
import logging
import skein
import os
import tempfile
import textwrap

import conda_pack

from mlflow.exceptions import ExecutionException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.entities import RunStatus
from mlflow.utils import file_utils

_logger = logging.getLogger(__name__)

YARN_APPLICATION_ID = "yarn_application_id"

# Configuration parameter names
YARN_NUM_CORES = "num_cores"
YARN_MEMORY = "memory"
YARN_QUEUE = "queue"
YARN_HADOOP_FILESYSTEMS = "hadoop_filesystems"
YARN_HADOOP_CONF_DIR = "hadoop_conf_dir"
# Extra parameters to configure skein setup
YARN_ENV = "env"
YARN_ADDITIONAL_FILES = "additional_files"

# default values for YARN related parameters
yarn_cfg_defaults = {
    YARN_NUM_CORES: 1,
    YARN_MEMORY: "1 GiB",
    YARN_QUEUE: "defautl",
    YARN_HADOOP_FILESYSTEMS: "",
    YARN_HADOOP_CONF_DIR: "",
    YARN_ENV: {},
    YARN_ADDITIONAL_FILES: [],
}


def run_yarn_job(
    remote_run,
    work_dir,
    parameters,
    experiment_id,
    cluster_spec,
    conda_env_name,
    entry_point_command,
    run_env_vars,
):
    env_params = _get_key_from_params(parameters, "env")
    additional_files = _get_key_from_params(parameters, "additional_files")

    yarn_config = _parse_yarn_config(cluster_spec, extra_params=parameters)

    env = _merge_env_lists(env_params, yarn_config[YARN_ENV])
    env.update(run_env_vars)
    additional_files += yarn_config[YARN_ADDITIONAL_FILES]
    zipped_project = _zip_project(work_dir)
    additional_files.append(zipped_project)
    packed_conda_env = _pack_conda_env(
        os.path.join(os.environ["MLFLOW_CONDA_HOME"], "envs", conda_env_name)
    )
    additional_files.append(packed_conda_env)

    try:
        with skein.Client() as skein_client:
            app_id = _submit(
                skein_client=skein_client,
                name="MLflow run for experiment {}".format(experiment_id),
                num_cores=yarn_config[YARN_NUM_CORES],
                memory=yarn_config[YARN_MEMORY],
                queue=yarn_config[YARN_QUEUE],
                hadoop_file_systems=yarn_config[YARN_HADOOP_FILESYSTEMS].split(","),
                hadoop_conf_dir=yarn_config[YARN_HADOOP_CONF_DIR],
                env_vars=env,
                additional_files=additional_files,
                entry_point_command=entry_point_command,
            )

            _logger.info("YARN backend launched app_id : %s", app_id)
            return YarnSubmittedRun(skein_app_id=app_id, mlflow_run_id=remote_run.info.run_id)
    finally:
        os.remove(zipped_project)
        os.remove(packed_conda_env)


def _submit(
    skein_client,
    entry_point_command,
    name="yarn_launcher",
    num_cores=1,
    memory="1 GiB",
    hadoop_file_systems=None,
    hadoop_conf_dir="",
    queue=None,
    env_vars=None,
    additional_files=None,
    node_label=None,
    num_containers=1,
    user=None,
):
    service = _generate_skein_service(
        memory,
        num_cores,
        num_containers,
        env_vars,
        additional_files,
        hadoop_conf_dir,
        entry_point_command,
    )

    spec = skein.ApplicationSpec(
        name=name,
        file_systems=hadoop_file_systems,
        services={name: service},
        acls=skein.model.ACLs(enable=True, ui_users=["*"], view_users=["*"]),
    )

    if user:
        spec.user = user

    if queue:
        spec.queue = queue

    if node_label:
        service.node_label = node_label

    return skein_client.submit(spec)


def _generate_skein_service(
    memory,
    num_cores,
    num_containers,
    env_vars,
    additional_files,
    hadoop_conf_dir,
    entry_point_command,
):
    env = dict(env_vars) if env_vars else dict()
    env.update({"SKEIN_CONFIG": "./.skein", "GIT_PYTHON_REFRESH": "quiet"})
    dict_files_to_upload = {
        os.path.basename(path): os.path.abspath(path) for path in additional_files
    }

    script = textwrap.dedent(
        """
                 set -x
                 env
                 export HADOOP_CONF_DIR=%s
                 export PATH=$(pwd)/conda_env.zip/bin:$PATH
                 export LD_LIBRARY_PATH=$(pwd)/conda_env.zip/lib
                 cd project.zip
                 %s
             """
        % (hadoop_conf_dir, entry_point_command)
    )

    return skein.Service(
        resources=skein.model.Resources(memory, num_cores),
        instances=num_containers,
        files=dict_files_to_upload,
        env=env,
        script=script,
    )


def _zip_project(project_dir):
    """
    Zip a project directory into an archive in a temp dir .

    :param project_dir: Path to a directory containing an MLflow project to upload to DBFS (e.g.
                        a directory containing an MLproject file).
    """
    temp_tarfile_dir = tempfile.mkdtemp()
    zip_filename = os.path.join(temp_tarfile_dir, "project.zip")
    file_utils.make_zipfile(zip_filename, project_dir)
    return zip_filename


def _pack_conda_env(conda_env_name):
    temp_tarfile_dir = tempfile.mkdtemp()
    conda_pack_filename = os.path.join(temp_tarfile_dir, "conda_env.zip")
    conda_pack.pack(prefix=conda_env_name, output=conda_pack_filename)
    return conda_pack_filename


def _validate_yarn_env(project):
    if not project.name:
        raise ExecutionException(
            "Project name in MLProject must be specified when " "using Yarn backend."
        )


def _merge_env_lists(env_params, env_yarn_cfg):
    env = dict(value.split("=") for value in env_params)
    env.update(dict(value.split("=") for value in env_yarn_cfg))
    return env


def _get_key_from_params(params, key, remove_key=True):
    if key not in params:
        return []

    values = params[key].split(",")
    if remove_key:
        del params[key]

    return values


def _parse_yarn_config(backend_config, extra_params=None):
    """
    Parses configuration for yarn backend and returns a dictionary
    with all needed values. In case values are not found in ``backend_config``
    dict passed, it is filled with the default values.
    """

    extra_params = extra_params or {}
    if not backend_config:
        raise ExecutionException("Backend_config file not found.")
    yarn_config = backend_config.copy()

    for cfg_key in [
        YARN_NUM_CORES,
        YARN_MEMORY,
        YARN_QUEUE,
        YARN_HADOOP_FILESYSTEMS,
        YARN_HADOOP_CONF_DIR,
        YARN_ENV,
        YARN_ADDITIONAL_FILES,
    ]:
        if cfg_key not in yarn_config.keys():
            yarn_config[cfg_key] = extra_params.get(cfg_key, yarn_cfg_defaults[cfg_key])
    return yarn_config


def _format_app_report(report):
    attrs = ["queue", "start_time", "finish_time", "final_status", "tracking_url", "user"]
    return os.linesep + os.linesep.join(
        "{:>16}: {}".format(attr, getattr(report, attr) or "") for attr in attrs
    )


def _get_application_logs(skein_client, app_id, wait_for_nb_logs=None, log_tries=15):
    for ind in range(log_tries):
        try:
            logs = skein_client.application_logs(app_id)
            nb_keys = len(logs.keys())
            _logger.info("Got %s/%s log files", nb_keys, wait_for_nb_logs)
            if not wait_for_nb_logs or nb_keys == wait_for_nb_logs:
                return logs
        except (skein.exceptions.ConnectionError, skein.exceptions.TimeoutError) as ex:
            _logger.exception(
                "Cannot collect logs (attempt %s/%s) due to exception: %s", ind + 1, log_tries, ex
            )
        time.sleep(3)
    return None


class YarnSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Yarn Job launched through skein to run an MLflow
    project.
    :param skein_app_id: ID of the submitted Skein Application.
    :param mlflow_run_id: ID of the MLflow project run.
    """

    POLL_STATUS_INTERNAL_SECS = 30

    def __init__(self, skein_app_id, mlflow_run_id):
        super(YarnSubmittedRun, self).__init__()
        self._skein_app_id = skein_app_id
        self._mlflow_run_id = mlflow_run_id

    @property
    def run_id(self):
        return self._mlflow_run_id

    def wait(self):
        status = skein.model.FinalStatus.UNDEFINED
        state = None

        with skein.Client() as skein_client:
            while True:
                app_report = skein_client.application_report(self._skein_app_id)
                if state != app_report.state:
                    _logger.info(_format_app_report(app_report))

                if app_report.final_status == skein.model.FinalStatus.FAILED:
                    _logger.info("YARN Application %s has failed", self._skein_app_id)

                if app_report.final_status != skein.model.FinalStatus.UNDEFINED:
                    break

                state = app_report.state
                time.sleep(self.POLL_STATUS_INTERNAL_SECS)

        return status == skein.model.FinalStatus.SUCCEEDED

    def cancel(self):
        with skein.Client() as skein_client:
            skein_client.kill_application(self._skein_app_id)

    def get_status(self):
        with skein.Client() as skein_client:
            app_report = skein_client.application_report(self._skein_app_id)
            return self._translate_to_runstate(app_report.state)

    def _translate_to_runstate(self, app_state):
        if app_state == skein.model.FinalStatus.SUCCEEDED:
            return RunStatus.FINISHED
        elif app_state == skein.model.FinalStatus.KILLED:
            return RunStatus.KILLED
        elif app_state == skein.model.FinalStatus.FAILED:
            return RunStatus.FAILED
        elif app_state == skein.model.FinalStatus.UNDEFINED:
            return RunStatus.RUNNING

        raise ExecutionException(
            "YARN Application %s has invalid status: %s" % (self._skein_app_id, app_state)
        )
