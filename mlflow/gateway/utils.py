import logging
import psutil
import re

from mlflow.exceptions import MlflowException


_logger = logging.getLogger(__name__)


def _parse_url_path_for_base_url(url_string):
    split_url = url_string.split("/")
    return "/".join(split_url[:-1])


def is_valid_endpoint_name(name: str) -> bool:
    """
    Check whether a string contains any URL reserved characters, spaces, or characters other
    than alphanumeric, underscore, and hyphen.

    Returns True if the string doesn't contain any of these characters.
    """
    return bool(re.fullmatch(r"^[\w-]+", name))


def check_configuration_route_name_collisions(config):
    if len(config) < 2:
        return
    names = [route["name"] for route in config]
    if len(names) != len(set(names)):
        raise MlflowException.invalid_parameter_value(
            "Duplicate names found in route configurations. Please remove the duplicate route "
            "name from the configuration to ensure that route endpoints are created properly."
        )


def kill_child_processes(parent_pid):
    """
    Gracefully terminate or kill child processes from a main process
    """
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):
        child.terminate()
    _, still_alive = psutil.wait_procs(parent.children(), timeout=3)
    for p in still_alive:
        p.kill()
