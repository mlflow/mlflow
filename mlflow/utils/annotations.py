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


def deprecated(alternative=None, since=None, impact=None):
    """
    Decorator for marking APIs deprecated in the docstring.

    :param func: A function to mark
    :returns Decorated function.
    """

    def deprecated_decorator(func):
        since_str = " since %s" % since if since else ""
        impact_str = impact if impact else "This method will be removed in a near future release."

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
