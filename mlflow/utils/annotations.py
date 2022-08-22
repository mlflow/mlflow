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

        notice = "``{function_name}`` is deprecated{since_string}. {impact}".format(
            function_name=".".join([func.__module__, func.__name__]),
            since_string=since_str,
            impact=impact_str,
        )
        if alternative is not None and alternative.strip():
            notice += " Use ``%s`` instead." % alternative

        @wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(notice, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        if func.__doc__ is not None:
            deprecated_func.__doc__ = ".. Warning:: " + notice + "\n" + func.__doc__

        return deprecated_func

    return deprecated_decorator


def deprecated_fields_included(
    deprecated_fields=None, alternative_fields=None, since=None, impact=None
):
    """
    Decorator for marking deprecated_fields of the class in the docstring.

    :param cls: A class to mark
    :returns Decorated class.
    """

    def deprecated_fields_included_decorator(cls):
        since_str = " since %s" % since if since else ""
        impact_str = impact if impact else ""

        if (
            deprecated_fields is not None
            and isinstance(deprecated_fields, list)
            and len(deprecated_fields) > 0
        ):
            impact_str = (
                impact if impact else "These fields ``%s`` will be deprecated." % deprecated_fields
            )

        notice = "Some fields of ``{cls_name}`` is deprecated{since_string}. {impact}".format(
            cls_name=cls.__class__.__name__,
            since_string=since_str,
            impact=impact_str,
        )

        if (
            alternative_fields is not None
            and isinstance(alternative_fields, list)
            and len(alternative_fields) > 0
        ):
            notice += " Use ``%s`` instead." % alternative_fields

        @wraps(cls)
        def deprecated_cls(*args, **kwargs):
            warnings.warn(notice, category=DeprecationWarning, stacklevel=2)
            return cls(*args, **kwargs)

        if cls.__doc__ is not None:
            deprecated_cls.__doc__ = ".. Warning:: " + notice + "\n" + cls.__doc__

        return deprecated_cls

    return deprecated_fields_included_decorator


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
