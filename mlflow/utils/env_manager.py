import warnings

LOCAL = "local"
NONE = "None"
CONDA = "conda"
VIRTUALENV = "virtualenv"


def validate(env_manager):
    allowed_values = [LOCAL, NONE, CONDA, VIRTUALENV]
    if env_manager not in allowed_values:
        raise ValueError(
            f"Invalid value for `env_manager`: {env_manager}. Must be one of {allowed_values}"
        )


def resolve(env_manager):
    validate(env_manager)
    if env_manager == LOCAL:
        warnings.warn(
            (
                "'local' option for `env_manager` is deprecated and will be removed in a future "
                "release. Use 'None' instead."
            ),
            UserWarning,
            stacklevel=2,
        )
        return NONE
    return env_manager
