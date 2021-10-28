import inspect


def _get_arg_names(f):
    """
    Get the argument names of a function.

    :param f: A function.
    :return: A list of argument names.
    """
    # `inspect.getargspec` or `inspect.getfullargspec` doesn't work properly for a wrapped function.
    # See https://hynek.me/articles/decorators#mangled-signatures for details.
    return list(inspect.signature(f).parameters.keys())
