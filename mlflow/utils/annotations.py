import inspect
import re
import types
import warnings
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

C = TypeVar("C", bound=Callable[..., Any])


def _get_min_indent_of_docstring(docstring_str: str) -> str:
    """
    Get the minimum indentation string of a docstring, based on the assumption
    that the closing triple quote for multiline comments must be on a new line.
    Note that based on ruff rule D209, the closing triple quote for multiline
    comments must be on a new line.

    Args:
        docstring_str: string with docstring

    Returns:
        Whitespace corresponding to the indent of a docstring.
    """

    if not docstring_str or "\n" not in docstring_str:
        return ""

    return re.match(r"^\s*", docstring_str.rsplit("\n", 1)[-1]).group()


def experimental(api_or_type: Union[C, str]) -> C:
    """Decorator / decorator creator for marking APIs experimental in the docstring.

    Args:
        api_or_type: An API to mark, or an API typestring for which to generate a decorator.

    Returns:
        Decorated API (if a ``api_or_type`` is an API) or a function that decorates
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
    indent = _get_min_indent_of_docstring(api.__doc__) if api.__doc__ else ""
    notice = (
        indent + f".. Note:: Experimental: This {api_type} may change or "
        "be removed in a future release without warning.\n\n"
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


_DEPRECATED_MARK_ATTR_NAME = "__deprecated"


def mark_deprecated(func):
    """
    Mark a function as deprecated by setting a private attribute on it.
    """
    setattr(func, _DEPRECATED_MARK_ATTR_NAME, True)


def is_marked_deprecated(func):
    """
    Is the function marked as deprecated.
    """
    return getattr(func, _DEPRECATED_MARK_ATTR_NAME, False)


def deprecated(
    alternative: Optional[str] = None, since: Optional[str] = None, impact: Optional[str] = None
):
    """Annotation decorator for marking APIs as deprecated in docstrings and raising a warning if
    called.

    Args:
        alternative: The name of a superseded replacement function, method,
            or class to use in place of the deprecated one.
        since: A version designator defining during which release the function,
            method, or class was marked as deprecated.
        impact: Indication of whether the method, function, or class will be
            removed in a future release.

    Returns:
        Decorated function or class.
    """

    def deprecated_decorator(obj):
        since_str = f" since {since}" if since else ""
        impact_str = impact if impact else "This method will be removed in a future release."

        qual_name = f"{obj.__module__}.{obj.__qualname__}"
        notice = f"``{qual_name}`` is deprecated{since_str}. {impact_str}"
        if alternative and alternative.strip():
            notice += f" Use ``{alternative}`` instead."

        if inspect.isclass(obj):
            original_init = obj.__init__

            @wraps(original_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(notice, category=FutureWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            obj.__init__ = new_init

            if obj.__doc__:
                obj.__doc__ = f".. Warning:: {notice}\n{obj.__doc__}"
            else:
                obj.__doc__ = f".. Warning:: {notice}"

            mark_deprecated(obj)
            return obj

        elif isinstance(obj, (types.FunctionType, types.MethodType)):

            @wraps(obj)
            def deprecated_func(*args, **kwargs):
                warnings.warn(notice, category=FutureWarning, stacklevel=2)
                return obj(*args, **kwargs)

            if obj.__doc__:
                indent = _get_min_indent_of_docstring(obj.__doc__)
                deprecated_func.__doc__ = f"{indent}.. Warning:: {notice}\n{obj.__doc__}"
            else:
                deprecated_func.__doc__ = f".. Warning:: {notice}"

            mark_deprecated(deprecated_func)
            return deprecated_func

        else:
            return obj

    return deprecated_decorator


def keyword_only(func):
    """A decorator that forces keyword arguments in the wrapped method."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            raise TypeError(f"Method {func.__name__} only takes keyword arguments.")
        return func(**kwargs)

    indent = _get_min_indent_of_docstring(wrapper.__doc__) if wrapper.__doc__ else ""
    notice = indent + ".. note:: This method requires all argument be specified by keyword.\n"
    wrapper.__doc__ = notice + wrapper.__doc__ if wrapper.__doc__ else notice

    return wrapper


def filter_user_warnings_once(func):
    """A decorator that filter user warnings to only show once in the wrapped method."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("once", category=UserWarning)
            return func(*args, **kwargs)

    return wrapper
