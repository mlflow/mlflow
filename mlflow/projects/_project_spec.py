"""Internal utilities for parsing MLproject YAML files."""

import os

from six.moves import shlex_quote

from mlflow import data


class Project(object):
    """A project specification loaded from an MLproject file."""
    def __init__(self, yaml_obj):
        self.conda_env = yaml_obj.get("conda_env")
        self.entry_points = {}
        for name, entry_point_yaml in yaml_obj.get("entry_points", {}).items():
            parameters = entry_point_yaml.get("parameters", {})
            command = entry_point_yaml.get("command")
            self.entry_points[name] = EntryPoint(name, parameters, command)
        # TODO: validate the spec, e.g. make sure paths in it are fine

    def get_entry_point(self, entry_point):
        from mlflow.projects import ExecutionException
        if entry_point in self.entry_points:
            return self.entry_points[entry_point]
        _, file_extension = os.path.splitext(entry_point)
        ext_to_cmd = {".py": "python", ".sh": os.environ.get("SHELL", "bash")}
        if file_extension in ext_to_cmd:
            command = "%s %s" % (ext_to_cmd[file_extension], shlex_quote(entry_point))
            return EntryPoint(name=entry_point, parameters={}, command=command)
        raise ExecutionException("Could not find {0} among entry points {1} or interpret {0} as a "
                                 "runnable script. Supported script file extensions: "
                                 "{2}".format(entry_point, list(self.entry_points.keys()),
                                              list(ext_to_cmd.keys())))


class EntryPoint(object):
    """An entry point in an MLproject specification."""
    def __init__(self, name, parameters, command):
        self.name = name
        self.parameters = {k: Parameter(k, v) for (k, v) in parameters.items()}
        self.command = command
        assert isinstance(self.command, str)

    def _validate_parameters(self, user_parameters):
        from mlflow.projects import ExecutionException
        missing_params = []
        for name in self.parameters:
            if (name not in user_parameters and self.parameters[name].default is None):
                missing_params.append(name)
        if len(missing_params) == 1:
            raise ExecutionException(
                "No value given for missing parameter: '%s'" % missing_params[0])
        elif len(missing_params) > 1:
            raise ExecutionException(
                "No value given for missing parameters: %s" %
                ", ".join(["'%s'" % name for name in missing_params]))

    def compute_parameters(self, user_parameters, storage_dir):
        """
        Given a dict mapping user-specified param names to values, computes parameters to
        substitute into the command for this entry point. Returns a tuple (params, extra_params)
        where `params` contains key-value pairs for parameters specified in the entry point
        definition, and `extra_params` contains key-value pairs for additional parameters passed
        by the user.

        Note that resolving parameter values can be a heavy operation, e.g. if a remote URI is
        passed for a parameter of type `path`, we download the URI to a local path within
        `storage_dir` and substitute in the local path as the parameter value.
        """
        if user_parameters is None:
            user_parameters = {}
        # Validate params before attempting to resolve parameter values
        self._validate_parameters(user_parameters)
        final_params = {}
        extra_params = {}

        for name, param_obj in self.parameters.items():
            if name in user_parameters:
                final_params[name] = param_obj.compute_value(user_parameters[name], storage_dir)
            else:
                final_params[name] = self.parameters[name].default
        for name in user_parameters:
            if name not in final_params:
                extra_params[name] = user_parameters[name]
        return self._sanitize_param_dict(final_params), self._sanitize_param_dict(extra_params)

    def compute_command(self, user_parameters, storage_dir):
        params, extra_params = self.compute_parameters(user_parameters, storage_dir)
        command_with_params = self.command.format(**params)
        command_arr = [command_with_params]
        command_arr.extend(["--%s %s" % (key, value) for key, value in extra_params.items()])
        return " ".join(command_arr)

    @staticmethod
    def _sanitize_param_dict(param_dict):
        return {str(key): shlex_quote(str(value)) for key, value in param_dict.items()}


class Parameter(object):
    """A parameter in an MLproject entry point."""
    def __init__(self, name, yaml_obj):
        self.name = name
        if isinstance(yaml_obj, str):
            self.type = yaml_obj
            self.default = None
        else:
            self.type = yaml_obj.get("type", "string")
            self.default = yaml_obj.get("default")

    def _compute_uri_value(self, user_param_value):
        from mlflow.projects import ExecutionException
        if not data.is_uri(user_param_value):
            raise ExecutionException("Expected URI for parameter %s but got "
                                     "%s" % (self.name, user_param_value))
        return user_param_value

    def _compute_path_value(self, user_param_value, storage_dir):
        from mlflow.projects import ExecutionException
        if not data.is_uri(user_param_value):
            if not os.path.exists(user_param_value):
                raise ExecutionException("Got value %s for parameter %s, but no such file or "
                                         "directory was found." % (user_param_value, self.name))
            return os.path.abspath(user_param_value)
        basename = os.path.basename(user_param_value)
        dest_path = os.path.join(storage_dir, basename)
        if dest_path != user_param_value:
            data.download_uri(uri=user_param_value, output_path=dest_path)
        return os.path.abspath(dest_path)

    def compute_value(self, user_param_value, storage_dir):
        if self.type != "path" and self.type != "uri":
            return user_param_value
        if self.type == "uri":
            return self._compute_uri_value(user_param_value)
        return self._compute_path_value(user_param_value, storage_dir)
