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


def param_logger(fn, args=[], kwargs=None, unlogged=[]):
    """
    Log parameters explicitly passed to a function.
    :param fn: function whose parameters are to be logged
    :param args: arguments explicitly passed into fn
    :param kwargs: kwargs explicitly passed into fn
    :param unlogged: parameters not to be logged
    :return: None
    """
    print("args len: " + str(len(args)))
    print("kwargs: " + str(kwargs))
    # Names of all parameters for the function
    all_param_names = inspect.getargspec(fn)[0]
    print("all_param_names: " + str(all_param_names))

    # Default values of all parameters with default values. Has length of n, and corresponds
    # to values of last n elements in all_param_names
    all_default_values = inspect.getargspec(fn)[3]
    print("all_default_values: " + str(all_default_values))

    # Checking if default values are present for logging. Known bug that getargspec will return an
    # empty argspec for certain functions, despite the functions having an argspec.
    if all_default_values:
        # Removing the first len(args) elements from all_param_names - these are values already
        # passed in explicitly by the user and don't need to be logged with default values.
        kwargs_and_default_names = all_param_names[len(args):]
        print("kwargs_and_default_names:" + str(kwargs_and_default_names))

        # If there are more parameter names than default values left, we know that the parameters
        # not covered by the default values are passed in as kwargs (assuming all non-default
        # parameters are passed to the function)
        if len(kwargs_and_default_names) > len(all_default_values):
            kwargs_and_default_names = kwargs_and_default_names[len(kwargs_and_default_names)
                                                                - len(all_default_values):]
        # Otherwise, if there are more default values than parameter names, we know that some of the
        # parameters with default values have been entered by the user in args
        elif len(kwargs_and_default_names) < len(all_default_values):
            all_default_values = all_default_values[len(all_default_values)
                                                    - len(kwargs_and_default_names):]

        default_params = zip(kwargs_and_default_names, all_default_values)

        # Filtering out the parameters that have been passed in by the user as a kwarg.
        default_params_to_be_logged = []
        for param in default_params:
            print(param[0])
            if param[0] not in kwargs:
                default_params_to_be_logged += [param]

        print("default params to be logged: " + str(default_params_to_be_logged))
        for param in default_params_to_be_logged:
            if param[0] not in unlogged:
                try_mlflow_log(mlflow.log_param, param[0], param[1])

    # List of tuples of parameter names and args that are passed by the user
    params_list = zip(inspect.getargspec(fn)[0][:len(args)], args)

    for param in params_list:
        if param[0] not in unlogged:
            try_mlflow_log(mlflow.log_param, param[0], param[1])

    for param_name in kwargs:
        if param_name not in unlogged:
            try_mlflow_log(mlflow.log_param, param_name, kwargs[param_name])
