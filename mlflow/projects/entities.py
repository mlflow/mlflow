"""
Contains core plugin-developer classes for representing MLflow projects, intended to simplify
implementing custom project execution backend plugins. For common
use cases like running MLflow projects, see :py:mod:`mlflow.projects`. For more information on
writing a plugin for a custom project execution backend, see `MLflow Plugins <../../plugins.html>`_.
"""

import os

from six.moves import shlex_quote

from mlflow import data
from mlflow.exceptions import ExecutionException
from mlflow.utils.file_utils import get_local_path_or_none
from mlflow.utils.string_utils import is_string_type


class Project(object):
    """
    A MLflow project specification
    """
    def __init__(self, entry_points, name, conda_env_path, docker_env):
        """
        Construct an in-memory representation of an MLflow project.

        :param entry_points: Dictionary mapping entry-point names (e.g. "main" or "training") to
                             :py:class:`mlflow.projects.entities.EntryPoint` objects.
        :param name: Name of the project, if any.
        :param conda_env_path: String path to conda.yaml file describing the conda environment to
                               use for executing the project, if any.
        :param docker_env: YAML object describing the docker image to use for executing the project,
                           if any. If provided, expected to contain an "image" field.
        """
        self.conda_env_path = conda_env_path
        self._entry_points = entry_points
        self.docker_env = docker_env
        self.name = name

    def get_entry_point(self, entry_point):
        """
        Returns the :py:class:`mlflow.projects.entities.EntryPoint` with the specified name
        within the current project.

        :param entry_point: Name of entry point
        :return: :py:class:`mlflow.projects.entities.EntryPoint` with the specified name
        """
        if entry_point in self._entry_points:
            return self._entry_points[entry_point]
        _, file_extension = os.path.splitext(entry_point)
        ext_to_cmd = {".py": "python", ".sh": os.environ.get("SHELL", "bash")}
        if file_extension in ext_to_cmd:
            command = "%s %s" % (ext_to_cmd[file_extension], shlex_quote(entry_point))
            if not is_string_type(command):
                command = command.encode("utf-8")
            return EntryPoint(name=entry_point, parameters={}, command=command)
        elif file_extension == ".R":
            command = "Rscript -e \"mlflow::mlflow_source('%s')\" --args" % shlex_quote(entry_point)
            return EntryPoint(name=entry_point, parameters={}, command=command)
        raise ExecutionException("Could not find {0} among entry points {1} or interpret {0} as a "
                                 "runnable script. Supported script file extensions: "
                                 "{2}".format(entry_point, list(self._entry_points.keys()),
                                              list(ext_to_cmd.keys())))


class EntryPoint(object):
    """
    An entry point in an MLproject specification.
    """
    def __init__(self, name, parameters, command):
        """
        Construct a new entry point.

        :param name: Entry point name, e.g. "main" or "training"
        :param parameters: Dictionary of parameter name to YAML object or string. If a YAML object,
                           should specify the parameter type and optionally a default value,
                           e.g. {"lr": {"default": 0.001, "type": "float"}}. If a string,
                           should specify the parameter type, e.g. "float", and the parameter is
                           assumed to have no default value (i.e. the user must specify a value
                           for the parameter in order to run the entry point).
        :param command: Templatized shell command to use for running the entry point, for example
                        "python train.py --learning-rate {lr}"
        """
        self.name = name
        self.parameters = {k: _Parameter(k, v) for (k, v) in parameters.items()}
        self.command = command

    def validate_parameters(self, user_parameters):
        """
        Validates that the provided params are sufficient to running the current entry point
        (i.e. that all required params have been specified), raising a
        :py:class:`mlflow.exceptions.ExecutionException` if not.

        :param user_parameters: Dictionary of param name to user-specified param value
        """
        missing_params = []
        for name in self.parameters:
            if (name not in user_parameters and self.parameters[name].default is None):
                missing_params.append(name)
        if missing_params:
            raise ExecutionException(
                "No value given for missing parameters: %s" %
                ", ".join(["'%s'" % name for name in missing_params]))

    def compute_parameters(self, user_parameters, storage_dir=None):
        """
        Computes parameters to substitute into the command for this entry point.

        Note that resolving parameter values can be an expensive operation, e.g. if a remote URI is
        passed for a parameter of type ``path``, we download the URI to a local path within
        ``storage_dir`` and substitute in the local path as the parameter value.

        :param user_parameters: dict mapping user-specified param names to values
        :param storage_dir: Local directory to which to download artifacts if a remote URI is
                            passed for a parameter of type "path". If unspecified, the remote
                            URI will be passed directly to the entry point command.
        :return: A tuple ``(params, extra_params)`` where ``params`` contains key-value pairs for
                 parameters specified in the entry point definition, and ``extra_params`` contains
                 key-value pairs for additional parameters passed by the user.
        """
        if user_parameters is None:
            user_parameters = {}
        # Validate params before attempting to resolve parameter values
        self.validate_parameters(user_parameters)
        final_params = {}
        extra_params = {}

        for key, param_obj in self.parameters.items():
            value = user_parameters[key] if key in user_parameters else self.parameters[key].default
            final_params[key] = param_obj.compute_value(value, storage_dir)
        for key in user_parameters:
            if key not in final_params:
                extra_params[key] = user_parameters[key]
        return self._sanitize_param_dict(final_params), self._sanitize_param_dict(extra_params)

    def compute_command(self, user_parameters, storage_dir=None):
        """
        Compute the shell command to run the current entry point

        :param user_parameters: Dictionary mapping param names to user-supplied param values
        :param storage_dir: Local directory to which to download artifacts if a remote URI is
                            passed for a parameter of type "path". If unspecified, the remote
                            URI will be passed directly to the entry point command.
        :return: String shell command, e.g. ``"python train.py --lr 0.001"``, to use for running the
                 entry point.
        """
        params, extra_params = self.compute_parameters(user_parameters, storage_dir=None)
        command_with_params = self.command.format(**params)
        command_arr = [command_with_params]
        command_arr.extend(["--%s %s" % (key, value) for key, value in extra_params.items()])
        return " ".join(command_arr)

    @staticmethod
    def _sanitize_param_dict(param_dict):
        return {str(key): shlex_quote(str(value)) for key, value in param_dict.items()}


class _Parameter(object):
    """A parameter in an MLproject entry point."""
    def __init__(self, name, yaml_obj):
        self.name = name
        if is_string_type(yaml_obj):
            self.type = yaml_obj
            self.default = None
        else:
            self.type = yaml_obj.get("type", "string")
            self.default = yaml_obj.get("default")

    def _compute_uri_value(self, user_param_value):
        if not data.is_uri(user_param_value):
            raise ExecutionException("Expected URI for parameter %s but got "
                                     "%s" % (self.name, user_param_value))
        return user_param_value

    def _compute_path_value(self, user_param_value, storage_dir):
        local_path = get_local_path_or_none(user_param_value)
        if local_path:
            if not os.path.exists(local_path):
                raise ExecutionException("Got value %s for parameter %s, but no such file or "
                                         "directory was found." % (user_param_value, self.name))
            return os.path.abspath(local_path)
        basename = os.path.basename(user_param_value)
        dest_path = os.path.join(storage_dir, basename)
        if dest_path != user_param_value:
            data.download_uri(uri=user_param_value, output_path=dest_path)
        return os.path.abspath(dest_path)

    def compute_value(self, param_value, storage_dir):
        if storage_dir and self.type == "path":
            return self._compute_path_value(param_value, storage_dir)
        elif self.type == "uri":
            return self._compute_uri_value(param_value)
        else:
            return param_value
