from mlflow.exceptions import ExecutionException
from mlflow.utils import process


def _validate_docker_env(project):
    if not project.name:
        raise ExecutionException("Project name in MLProject must be specified when using docker "
                                 "for image tagging.")
    if not project.docker_env.get('image'):
        raise ExecutionException("Project with docker environment must specify the docker image "
                                 "to use via an 'image' field under the 'docker_env' field.")


def _validate_docker_installation():
    """
    Verify if Docker is installed on host machine.
    """
    try:
        docker_path = "docker"
        process.exec_cmd([docker_path, "--help"], throw_on_error=False)
    except EnvironmentError:
        raise ExecutionException("Could not find Docker executable. "
                                 "Ensure Docker is installed as per the instructions "
                                 "at https://docs.docker.com/install/overview/.")
