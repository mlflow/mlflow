import os


def get_env(variable_name, default=None):
    """
    :param variable_name: The name of the environment variable to retrieve.
    :param default: The value to return in the case that the environment variable is not defined.
    """
    return os.environ.get(variable_name, failobj=default)


def unset_variable(variable_name):
    if variable_name in os.environ:
        del os.environ[variable_name]
