import inspect
import warnings
from functools import wraps


def experimental(func):
    """
    Decorator for marking APIs experimental in the docstring.

    :param func: A function to mark
    :returns Decorated function.
    """
    notice = (
        "    .. Note:: Experimental: This method may change or "
        + "be removed in a future release without warning.\n\n"
    )
    func.__doc__ = notice + func.__doc__
    return func


def deprecated(alternative=None, since=None):
    """
    Decorator for marking APIs deprecated in the docstring.

    :param func: A function to mark
    :returns Decorated function.
    """

    def deprecated_decorator(func):
        since_str = " since %s" % since if since else ""
        notice = (
            ".. Warning:: ``{function_name}`` is deprecated{since_string}. This method will be"
            " removed in a near future release.".format(
                function_name=".".join([func.__module__, func.__name__]), since_string=since_str
            )
        )
        if alternative is not None and alternative.strip():
            notice += " Use ``%s`` instead." % alternative

        @wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(notice, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        if func.__doc__ is not None:
            deprecated_func.__doc__ = notice + "\n" + func.__doc__

        return deprecated_func

    return deprecated_decorator


def keyword_only(func):
    """
    A decorator that forces keyword arguments in the wrapped method.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            raise TypeError("Method %s only takes keyword arguments." % func.__name__)
        return func(**kwargs)

    notice = ".. Note:: This method requires all argument be specified by keyword.\n"
    wrapper.__doc__ = notice + wrapper.__doc__
    return wrapper


def deprecate_conda_env(func):
    """
    Wraps the given function to raise a deprecation warning when the `conda_env` argument is
    supplied.
    """
    conda_env_var_name = "conda_env"
    parameters = inspect.signature(func).parameters
    conda_env_index = list(parameters.keys()).index(conda_env_var_name)
    # Assuming `conda_env` is defined as a keyword or keyword-only argument
    default_value = parameters[conda_env_var_name].default

    @wraps(func)
    def wrapper(*args, **kwargs):
        conda_env_val = (
            args[conda_env_index]
            if len(args) > conda_env_index
            else kwargs.get(conda_env_var_name, default_value)
        )
        if conda_env_val != default_value:
            warnings.warn(
                (
                    "`conda_env` has been deprecated, please use `pip_requirements` or "
                    "`additional_pip_requirements` instead."
                ),
                FutureWarning,
                stacklevel=2,
            )
        return func(*args, **kwargs)

    return wrapper
