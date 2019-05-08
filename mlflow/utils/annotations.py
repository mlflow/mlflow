def experimental(func):
    """
    Decorator for marking APIs experimental in the docstring.

    :param func: A function to mark
    :returns Decorated function.
    """
    notice = ".. Note:: Experimental: This method may change or " + \
             "be removed in a future release without warning.\n"
    func.__doc__ = notice + func.__doc__
    return func


def deprecated(alternative=None):
    """
    Decorator for marking APIs deprecated in the docstring.

    :param func: A function to mark
    :returns Decorated function.
    """
    def deprecated_func(func):
        notice = ".. Warning:: Deprecated: This method will be deprecated in " + \
                 "a near future release."
        if alternative is not None and alternative.strip():
            notice += " Use ``%s`` instead." % alternative
        func.__doc__ = notice + "\n" + func.__doc__
        return func
    return deprecated_func
