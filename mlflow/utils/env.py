import os


def get_env(variable_name):
    return os.environ.get(variable_name)


def unset_variable(variable_name):
    if variable_name in os.environ:
        del os.environ[variable_name]


class EnvironmentVariable:
    def __init__(self, name, type, default):
        self.name = name
        self.type = type
        self.default = default

    def get(self):
        val = os.getenv(self.name)
        if val:
            try:
                return self.type(val)
            except Exception as e:
                raise ValueError(
                    f"Parse environment config {self.name}'s value '{val}' failed. (error: {repr(e)})"
                )
        return self.default

    def __str__(self):
        return f"Environment variable: name={self.name}, type={self.type}, default={self.default}"
