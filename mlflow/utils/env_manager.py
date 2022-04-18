LOCAL = "local"
CONDA = "conda"
VIRTUALENV = "virtualenv"


def validate(env_manager):
    allowed_values = [LOCAL, CONDA, VIRTUALENV]
    if env_manager not in allowed_values:
        raise ValueError(
            f"Invalid value for `env_manager`: {env_manager}. Must be one of {allowed_values}"
        )
