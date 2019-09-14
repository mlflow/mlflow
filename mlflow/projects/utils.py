import os
import logging

from mlflow.exceptions import ExecutionException

_logger = logging.getLogger(__name__)


MLPROJECT_PARAMETER_TYPES = ('string', 'float', 'path', 'uri')
BAD_MLPROJECT_MESSAGE = "Invalid MLproject file: {}"


def validate_conda_env_path(path):
    if not os.path.exists(path):
        bad_conda_path_message = "conda environment file {} not found".format(path)
        raise ExecutionException(
            BAD_MLPROJECT_MESSAGE.format(bad_conda_path_message)
        )


def validate_project_yaml(project_yaml):
    """validate the yaml in an MLproject file"""
    project_allowed_entries = ('name', 'conda_env', 'docker_env', 'entry_points')

    project_entries = project_yaml.keys()

    # make sure there are no extraneous entries
    _validate_entries_are_allowed('project-level', project_entries, project_allowed_entries)

    if 'name' in project_entries:
        _validate_name_yaml(project_yaml.get('name'))

    if 'docker_env' in project_entries:
        _validate_docker_env_yaml(project_yaml.get('docker_env'))
        # disallow conda and docker envs together in the same MLproject file
        if 'conda_env' in project_entries:
            multi_env_message = "cannot specify conda_env and docker_env in the same project"
            raise ExecutionException(BAD_MLPROJECT_MESSAGE.format(multi_env_message))

    if 'conda_env' in project_entries:
        _validate_conda_env_yaml(project_yaml.get('conda_env'))

    if 'entry_points' in project_entries:
        for name, entry_point in project_yaml.get('entry_points').items():
            _validate_entry_point_yaml(name, entry_point)


def _validate_entries_are_allowed(yaml_level, present_entries, allowed_entries):
    """helper to validate that entry keys are in an allowable set"""
    for k in present_entries:
        if k not in allowed_entries:
            message = "{} entry {} not one of allowed entries {{{}}}"\
                .format(yaml_level, k, ', '.join(allowed_entries))
            raise ExecutionException(BAD_MLPROJECT_MESSAGE.format(message))


def _validate_entry_point_yaml(entry_point_name, entry_point_yaml):
    """helper to validate entry point yaml"""
    entry_point_allowed_entries = ('parameters', 'command')

    entry_point_entries = entry_point_yaml.keys()
    _validate_entries_are_allowed('entry point', entry_point_entries, entry_point_allowed_entries)

    if 'command' not in entry_point_entries:
        no_command_message = "entry point {} has no corresponding command".format(entry_point_name)
        raise ExecutionException(BAD_MLPROJECT_MESSAGE.format(no_command_message))

    if 'parameters' in entry_point_entries:
        for parameter_name, parameter_yaml in entry_point_yaml.get('parameters').items():
            _validate_entry_point_parameter_yaml(entry_point_name, parameter_name, parameter_yaml)


def _validate_entry_point_parameter_yaml(entry_point_name, parameter_name, parameter_yaml):
    """helper to validate parameter yaml within an entry point"""
    parameter_allowed_entries = ('type', 'default')

    # define this here so we can provide a more informative message
    def check_parameter_type(attempted_type):
        if attempted_type not in MLPROJECT_PARAMETER_TYPES:
            unsupported_parameter_type_message = (
                "unsupported type for parameter {} in entry point {}, "
                "parameter value will be converted to string"
            ).format(parameter_name, entry_point_name)
            _logger.warning(unsupported_parameter_type_message)

    # interpret the entry as a type if it's a single string
    if isinstance(parameter_yaml, str):
        check_parameter_type(parameter_yaml)
        return

    # otherwise expect sub-entries, at least for 'type', potentially also for 'default'
    parameter_entries = parameter_yaml.keys()
    _validate_entries_are_allowed('parameter', parameter_entries, parameter_allowed_entries)

    if 'type' not in parameter_entries:
        missing_parameter_type_message = "parameter {} in entry point {} must specify a type" \
            .format(parameter_name, entry_point_name)
        raise ExecutionException(BAD_MLPROJECT_MESSAGE.format(missing_parameter_type_message))

    # TODO: validate 'default' parameter value somehow
    if 'default' in parameter_entries:
        pass

    check_parameter_type(parameter_yaml.get('type'))


def _validate_docker_env_yaml(docker_env_yaml):

    if not isinstance(docker_env_yaml, dict) or not isinstance(docker_env_yaml.get('image'), str):
        bad_docker_entry_message = (
            "docker_env must be a YAML object with a string 'image' entry representing "
            "a Docker image that is accessible on the system executing the project"
        )
        raise ExecutionException(BAD_MLPROJECT_MESSAGE.format(bad_docker_entry_message))

    docker_env_entries = docker_env_yaml.keys()
    _validate_entries_are_allowed('docker env', docker_env_entries, ('image',))


def _validate_conda_env_yaml(conda_env_yaml):
    # TODO: make this condition more specific
    if not isinstance(conda_env_yaml, str):
        bad_conda_env_message = (
            "conda_env should be a string representing the relative path to a "
            "Conda environment YAML file in the MLflow project's directory"
        )
        raise ExecutionException(BAD_MLPROJECT_MESSAGE.format(bad_conda_env_message))


def _validate_name_yaml(name_yaml):
    if not isinstance(name_yaml, str):
        raise ExecutionException(BAD_MLPROJECT_MESSAGE.format("name must be a single string"))
