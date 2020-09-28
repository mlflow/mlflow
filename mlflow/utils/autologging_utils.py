import inspect
import functools
import gorilla
import warnings

import mlflow


INPUT_EXAMPLE_SAMPLE_ROWS = 5
FAILED_INPUT_EXAMPLE_PREFIX_TEXT = "Failed to gather input example: "
FAILED_MODEL_SIGNATURE_PREFIX_TEXT = "Failed to infer model signature: "
ENSURE_AUTOLOGGING_ENABLED_TEXT = (
    "please ensure that autologging is enabled before constructing the dataset."
)


def try_mlflow_log(fn, *args, **kwargs):
    """
    Catch exceptions and log a warning to avoid autolog throwing.
    """
    try:
        fn(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        warnings.warn("Logging to MLflow failed: " + str(e), stacklevel=2)


def get_unspecified_default_args(user_args, user_kwargs, all_param_names, all_default_values):
    """
    Determine which default values are used in a call, given args and kwargs that are passed in.

    :param user_args: list of arguments passed in by the user
    :param user_kwargs: dictionary of kwargs passed in by the user
    :param all_param_names: names of all of the parameters of the function
    :param all_default_values: values of all default parameters
    :return: a dictionary mapping arguments not specified by the user -> default value
    """
    num_args_without_default_value = len(all_param_names) - len(all_default_values)

    # all_default_values correspond to the last len(all_default_values) elements of the arguments
    default_param_names = all_param_names[num_args_without_default_value:]

    default_args = dict(zip(default_param_names, all_default_values))

    # The set of keyword arguments that should not be logged with default values
    user_specified_arg_names = set(user_kwargs.keys())

    num_user_args = len(user_args)

    # This checks if the user passed values for arguments with default values
    if num_user_args > num_args_without_default_value:
        num_default_args_passed_as_positional = num_user_args - num_args_without_default_value
        # Adding the set of positional arguments that should not be logged with default values
        names_to_exclude = default_param_names[:num_default_args_passed_as_positional]
        user_specified_arg_names.update(names_to_exclude)

    return {
        name: value for name, value in default_args.items() if name not in user_specified_arg_names
    }


def log_fn_args_as_params(fn, args, kwargs, unlogged=[]):  # pylint: disable=W0102
    """
    Log parameters explicitly passed to a function.
    :param fn: function whose parameters are to be logged
    :param args: arguments explicitly passed into fn
    :param kwargs: kwargs explicitly passed into fn
    :param unlogged: parameters not to be logged
    :return: None
    """
    # all_default_values has length n, corresponding to values of the
    # last n elements in all_param_names
    pos_params, _, _, pos_defaults, kw_params, kw_defaults, _ = inspect.getfullargspec(fn)

    kw_params = list(kw_params) if kw_params else []
    pos_defaults = list(pos_defaults) if pos_defaults else []
    all_param_names = pos_params + kw_params
    all_default_values = pos_defaults + [kw_defaults[param] for param in kw_params]

    # Checking if default values are present for logging. Known bug that getargspec will return an
    # empty argspec for certain functions, despite the functions having an argspec.
    if all_default_values is not None and len(all_default_values) > 0:
        # Logging the default arguments not passed by the user
        defaults = get_unspecified_default_args(args, kwargs, all_param_names, all_default_values)

        for name in [name for name in defaults.keys() if name in unlogged]:
            del defaults[name]
        try_mlflow_log(mlflow.log_params, defaults)

    # Logging the arguments passed by the user
    args_dict = dict(
        (param_name, param_val)
        for param_name, param_val in zip(all_param_names, args)
        if param_name not in unlogged
    )

    if args_dict:
        try_mlflow_log(mlflow.log_params, args_dict)

    # Logging the kwargs passed by the user
    for param_name in kwargs:
        if param_name not in unlogged:
            try_mlflow_log(mlflow.log_param, param_name, kwargs[param_name])


def wrap_patch(destination, name, patch, settings=None):
    """
    Apply a patch while preserving the attributes (e.g. __doc__) of an original function.

    :param destination: Patch destination
    :param name: Name of the attribute at the destination
    :param patch: Patch function
    :param settings: Settings for gorilla.Patch
    """
    if settings is None:
        settings = gorilla.Settings(allow_hit=True, store_hit=True)

    original = getattr(destination, name)
    wrapped = functools.wraps(original)(patch)
    patch = gorilla.Patch(destination, name, wrapped, settings=settings)
    gorilla.apply(patch)


class _InputExampleInfo:
    def __init__(self, input_example=None, error_msg=None):
        self.input_example = input_example
        self.error_msg = error_msg

def handle_input_example_and_signature(get_input_example, infer_model_signature, log_input_example, log_model_signature, logger):
    input_example = None
    input_example_user_msg = None
    input_example_failure_msg = None
    if log_input_example or log_model_signature:
        try:
            input_example = get_input_example()
        except Exception as e:  # pylint: disable=broad-except
            input_example_failure_msg = str(e)
            input_example_user_msg = FAILED_INPUT_EXAMPLE_PREFIX_TEXT + str(e)

    model_signature = None
    model_signature_user_msg = None
    if log_model_signature:
        try:
            if input_example is None:
                raise Exception("could not sample data to infer model signature: " + input_example_failure_msg)
            model_signature = infer_model_signature(input_example)
        except Exception as e:  # pylint: disable=broad-except
            model_signature_user_msg = FAILED_MODEL_SIGNATURE_PREFIX_TEXT + str(e)

    if log_input_example and input_example_user_msg is not None:
        logger.warning(input_example_user_msg)
    if log_model_signature and model_signature_user_msg is not None:
        logger.warning(model_signature_user_msg)

    return input_example if log_input_example else None, model_signature