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


def param_logger(original, args, kwargs, unlogged=[]):
    """
    Log parameters explicitly passed to a function.
    :param original: function whose parameters are to be logged
    :param args: arguments passed into the original function
    :param kwargs: kwargs passed into the original function
    :param unlogged: parameters not to be logged
    :return: None
    """

    # List of tuples of parameter names and args that are passed by the user
    params_list = zip(inspect.getfullargspec(original)[0][:len(args)], args)

    for param in params_list:
        if param[0] not in unlogged:
            try_mlflow_log(mlflow.log_param, param[0], param[1])

    for param_name in kwargs:
        if param_name not in unlogged:
            try_mlflow_log(mlflow.log_param, param_name, kwargs[param_name])
