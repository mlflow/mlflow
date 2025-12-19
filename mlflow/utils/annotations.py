import inspect
import re
import types
import warnings
from functools import wraps
from typing import Callable, ParamSpec, TypeVar, overload


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


P = ParamSpec("P")
R = TypeVar("R")


@overload
def experimental(
    f: Callable[P, R],
    version: str | None = None,
) -> Callable[P, R]: ...


@overload
def experimental(
    f: None = None,
    version: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def experimental(
    f: Callable[P, R] | None = None,
    version: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator / decorator creator for marking APIs experimental in the docstring.

    Args:
        f: The function to be decorated.
        version: The version in which the API was introduced as experimental.
            The version is used to determine whether the API should be considered
            as stable or not when releasing a new version of MLflow.

    Returns:
        A decorator that adds a note to the docstring of the decorated API,
    """
    if f:
        return _experimental(f)
    else:

        def decorator(f: Callable[P, R]) -> Callable[P, R]:
            return _experimental(f)

        return decorator


def _experimental(api: Callable[P, R]) -> Callable[P, R]:
    if inspect.isclass(api):
        api_type = "class"
    elif inspect.isfunction(api):
        api_type = "function"
    elif isinstance(api, (property, types.MethodType)):
        api_type = "property"
    else:
        api_type = str(type(api))

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


def deprecated(alternative: str | None = None, since: str | None = None, impact: str | None = None):
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
        impact_str = impact or "This method will be removed in a future release."

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


def deprecated_parameter(old_param: str, new_param: str, version: str | None = None):
    """
    Decorator to handle deprecated parameter renaming with automatic warning and forwarding.

    This decorator:
    1. Accepts the deprecated parameter in the function signature
    2. Emits a deprecation warning when the old parameter is used
    3. Maps the old parameter value to the new parameter name
    4. Hides the deprecated parameter from documentation
    5. Keeps the function body untouched

    Args:
        old_param: The deprecated parameter name
        new_param: The new parameter name to use instead
        version: Optional version when the deprecation will be removed

    Example:
        @deprecated_parameter("request_id", "trace_id", version="4.0.0")
        def search_traces(trace_id: str | None = None):
            # Function body uses trace_id directly
            return trace_id

        # Users can still call with old parameter:
        search_traces(request_id="123")  # Issues warning, forwards to trace_id
        search_traces(trace_id="123")    # No warning
    """

    def decorator(func):
        sig = inspect.signature(func)
        params = dict(sig.parameters)

        if new_param not in params:
            raise ValueError(
                f"New parameter '{new_param}' not found in function '{func.__name__}' signature"
            )

        @wraps(func)
        def wrapper(*args, **kwargs):
            if old_param in kwargs:
                old_value = kwargs.pop(old_param)

                version_msg = f" and will be removed in version {version}" if version else ""
                warnings.warn(
                    f"Parameter '{old_param}' is deprecated{version_msg}. "
                    f"Please use '{new_param}' instead.",
                    category=FutureWarning,
                    stacklevel=2,
                )

                if new_param in kwargs:
                    raise ValueError(
                        f"Cannot specify both '{old_param}' (deprecated) and '{new_param}'. "
                        f"Use '{new_param}' only."
                    )

                kwargs[new_param] = old_value

            return func(*args, **kwargs)

        # Update the wrapper's signature to include the deprecated parameter as keyword-only
        # but keep it out of documentation
        wrapper.__signature__ = sig

        # Update docstring to note the deprecation (if docstring exists)
        if func.__doc__:
            indent = _get_min_indent_of_docstring(func.__doc__)
            deprecation_note = (
                f"{indent}.. Note:: Parameter ``{old_param}`` is deprecated. "
                f"Use ``{new_param}`` instead."
            )
            wrapper.__doc__ = f"{deprecation_note}\n{func.__doc__}"

        return wrapper

    return decorator


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


def requires_sql_backend(func):
    """
    Decorator for marking APIs that require a SQL-based tracking backend.

    This decorator:
    1. Adds a note to the docstring indicating SQL backend requirement
    2. When used with FileStore, raises a helpful error message

    The decorator should be applied to methods in AbstractStore that are only
    implemented in SQL-based backends (SQLAlchemyStore, RestStore).
    """
    indent = _get_min_indent_of_docstring(func.__doc__) if func.__doc__ else ""
    notice = (
        indent + ".. Note:: This method requires a SQL-based tracking backend "
        "(e.g., SQLite, PostgreSQL, MySQL). It is not supported with FileStore.\n\n"
    )
    func.__doc__ = notice + func.__doc__ if func.__doc__ else notice

    func._requires_sql_backend = True

    return func
