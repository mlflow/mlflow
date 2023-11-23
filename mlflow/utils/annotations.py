import inspect
import types
import warnings
from functools import wraps
from typing import Any, Callable, TypeVar, Union

C = TypeVar("C", bound=Callable[..., Any])


def experimental(api_or_type: Union[C, str]) -> C:
    """
    Decorator / decorator creator for marking APIs experimental in the docstring.

    :param api_or_type: An API to mark, or an API typestring for which to generate a decorator.
    :return: Decorated API (if a ``api_or_type`` is an API) or a function that decorates
             the specified API type (if ``api_or_type`` is a typestring).
    """
    if isinstance(api_or_type, str):

        def f(api: C) -> C:
            return _experimental(api=api, api_type=api_or_type)

        return f
    elif inspect.isclass(api_or_type):
        return _experimental(api=api_or_type, api_type="class")
    elif inspect.isfunction(api_or_type):
        return _experimental(api=api_or_type, api_type="function")
    elif isinstance(api_or_type, (property, types.MethodType)):
        return _experimental(api=api_or_type, api_type="property")
    else:
        return _experimental(api=api_or_type, api_type=str(type(api_or_type)))


def _experimental(api: C, api_type: str) -> C:
    notice = (
        f"    .. Note:: Experimental: This {api_type} may change or "
        + "be removed in a future release without warning.\n\n"
    )
    if api_type == "property":
        api.__doc__ = api.__doc__ + "\n\n" + notice if api.__doc__ else notice
    else:
        api.__doc__ = notice + api.__doc__ if api.__doc__ else notice
    return api


def developer_stable(func):
    """
    The API marked here as `@developer_stable` has certain protections associated with future
    development work.
    Classes marked with this decorator implicitly apply this status to all methods contained within
    them.

    APIs that are annotated with this decorator are guaranteed (except in cases of notes below) to:
    - maintain backwards compatibility such that earlier versions of any MLflow client, cli, or
      server will not have issues with any changes being made to them from an interface perspective.
    - maintain a consistent contract with respect to existing named arguments such that
      modifications will not alter or remove an existing named argument.
    - maintain implied or declared types of arguments within its signature.
    - maintain consistent behavior with respect to return types.

    Note: Should an API marked as `@developer_stable` require a modification for enhanced feature
      functionality, a deprecation warning will be added to the API well in advance of its
      modification.

    Note: Should an API marked as `@developer_stable` require patching for any security reason,
      advanced notice is not guaranteed and the labeling of such API as stable will be ignored
      for the sake of such a security patch.

    """
    return func


def deprecated(alternative=None, since=None, impact=None):
    """
    Annotation decorator for marking APIs as deprecated in docstrings and raising a warning if
    called.
    :param alternative: (Optional string) The name of a superseded replacement function, method,
                        or class to use in place of the deprecated one.
    :param since: (Optional string) A version designator defining during which release the function,
                  method, or class was marked as deprecated.
    :param impact: (Optional boolean) Indication of whether the method, function, or class will be
                   removed in a future release.
    :return: Decorated function.
    """

    def deprecated_decorator(func):
        since_str = f" since {since}" if since else ""
        impact_str = impact if impact else "This method will be removed in a future release."

        notice = "``{qual_function_name}`` is deprecated{since_string}. {impact}".format(
            qual_function_name=".".join([func.__module__, func.__qualname__]),
            since_string=since_str,
            impact=impact_str,
        )
        if alternative is not None and alternative.strip():
            notice += f" Use ``{alternative}`` instead."

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
            raise TypeError(f"Method {func.__name__} only takes keyword arguments.")
        return func(**kwargs)

    notice = ".. Note:: This method requires all argument be specified by keyword.\n"
    wrapper.__doc__ = notice + wrapper.__doc__ if wrapper.__doc__ else notice
    return wrapper
