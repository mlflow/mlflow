import os

# Necessary workaround for this issue:
# https://github.com/google/protobuf/issues/3002#issuecomment-325459597
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# pylint: disable=wrong-import-position
import mlflow.projects as projects  # noqa
import mlflow.tracking as tracking  # noqa

log_param = tracking.log_param
log_metric = tracking.log_metric
log_artifacts = tracking.log_artifacts
log_artifact = tracking.log_artifact
active_run = tracking.active_run
start_run = tracking.start_run
end_run = tracking.end_run
get_artifact_uri = tracking.get_artifact_uri
set_tracking_uri = tracking.set_tracking_uri
get_tracking_uri = tracking.get_tracking_uri
create_experiment = tracking.create_experiment

run = projects.run

__all__ = ["log_param", "log_metric", "log_artifacts", "log_artifact", "active_run",
           "start_run", "end_run", "get_artifact_uri", "set_tracking_uri", "create_experiment"]


def _dev_mode():
    """
    Internal function to determine whether we are running in developer mode.

    :return: True if running in developer mode.
    """
    return "MLFLOW_DEV" in os.environ


def _root_dir():
    """
    Internal function to retrieve root directoryof local dev copy of mlflow project.
    Used during development e.g. when we need to copy most recent files into a docker container.

    :return: mlflow path
    """
    assert _dev_mode()
    if "MLFLOW_HOME" in os.environ:
        return os.environ.get("MLFLOW_HOME")
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _copy_mlflow_project(output_dir=""):
    """
    Internal function used to copy mlflow project during development.
    The mlflow is assumed to be accessible as a local directory in this case.

    :param output_dir: mlflow will be copied here
    :return: name of the mlflow project directory
    """

    def _docker_ignore(mlflow_root):
        docker_ignore = os.path.join(mlflow_root, '.dockerignore')
        patterns = []
        if os.path.exists(docker_ignore):
            with open(docker_ignore, "r") as f:
                patterns = [x.strip() for x in f.readlines()]

        def ignore(_, names):
            import fnmatch
            res = set()
            for p in patterns:
                res.update(set(fnmatch.filter(names, p)))
            return list(res)

        return ignore if patterns else None

    mlflow_dir = "mlflow-project"
    mlflow_root = _root_dir()
    import shutil
    shutil.copytree(mlflow_root, os.path.join(output_dir, mlflow_dir),
                    ignore=_docker_ignore(mlflow_root))
    return mlflow_dir
