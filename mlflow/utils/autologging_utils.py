import inspect
import mlflow
import warnings


def try_mlflow_log(fn, *args, **kwargs):
    """
    Catch exceptions and log a warning to avoid autolog throwing.
    """
    try:
        fn(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        warnings.warn("Logging to MLflow failed: " + str(e), stacklevel=2)


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
    all_param_names, _, _, all_default_values = inspect.getargspec(fn)  # pylint: disable=W1505

    # Checking if default values are present for logging. Known bug that getargspec will return an
    # empty argspec for certain functions, despite the functions having an argspec.
    if all_default_values is not None and len(all_default_values) > 0:

        # Now, compute the names of all params that were not supplied by the user (e.g. all params
        # for which we use the default value). Start by removing the first len(args) elements from 
        # all_param_names - these are names of params passed as positional arguments
        # by the user and don't need to be logged with default values.
        kwargs_and_default_names = all_param_names[len(args):]

        # If there are more parameter names than default values left, we know that the parameters
        # not covered by the default values were passed by the user as kwargs (assuming all non-default
        # parameters are passed to the function)
        if len(kwargs_and_default_names) > len(all_default_values):
            kwargs_and_default_names = kwargs_and_default_names[len(kwargs_and_default_names)
                                                                - len(all_default_values):]
        # Otherwise, if there are more default values than parameter names, we know that some of the
        # parameters with default values have been entered by the user in args
        elif len(kwargs_and_default_names) < len(all_default_values):
            all_default_values = all_default_values[len(all_default_values)
                                                    - len(kwargs_and_default_names):]

        # Filtering out the parameters that have been passed in by the user as a kwarg.
        default_param_names = set(kwargs_and_default_names) - set(kwargs.keys())

        default_params = dict(zip(kwargs_and_default_names, all_default_values))

        for name in default_param_names:
            if name not in unlogged:
                try_mlflow_log(mlflow.log_param, name, default_params[name])

    # Logging the arguments passed by the user
    args_dict = dict((param_name, param_val) for param_name, param_val
                     in zip(all_param_names[:len(args)], args)
                     if param_name not in unlogged)

    try_mlflow_log(mlflow.log_params, args_dict)

    # Logging the kwargs passed by the user
    for param_name in kwargs:
        if param_name not in unlogged:
            try_mlflow_log(mlflow.log_param, param_name, kwargs[param_name])
