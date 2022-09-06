import inspect
import warnings
from functools import wraps
from typing import Any, Union


def experimental(api_or_type: Union[callable, str]):
    """
    Decorator / decorator creator for marking APIs experimental in the docstring.

    :param api: An API to mark, or an API typestring for which to generate a decorator.
    :return: Decorated API (if a ``api_or_type`` is an API) or a function that decorates
             the specified API type (if ``api_or_type`` is a typestring).
    """
    if isinstance(api_or_type, str):
        return lambda api: _experimental(api=api, api_type=api_or_type)
    elif inspect.isclass(api_or_type):
        return _experimental(api=api_or_type, api_type="class")
    else:
        return _experimental(api=api_or_type, api_type="method")


def _experimental(api: Any, api_type: str):
    notice = (
        f"    .. Note:: Experimental: This {api_type} may change or "
        + "be removed in a future release without warning.\n\n"
    )
    api.__doc__ = notice + api.__doc__
    return api


def deprecated(alternative=None, since=None, impact=None):
    """
    Decorator for marking APIs deprecated in the docstring.
    :param func: A function to mark
    :returns Decorated function.
    """

    def deprecated_decorator(func):
        since_str = " since %s" % since if since else ""
        impact_str = impact if impact else "This method will be removed in a future release."

        notice = "``{qual_function_name}`` is deprecated{since_string}. {impact}".format(
            qual_function_name=".".join([func.__module__, func.__qualname__]),
            since_string=since_str,
            impact=impact_str,
        )
        if alternative is not None and alternative.strip():
            notice += " Use ``%s`` instead." % alternative

        @wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(notice, category=FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        if func.__doc__ is not None:
            deprecated_func.__doc__ = ".. Warning:: " + notice + "\n" + func.__doc__

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
