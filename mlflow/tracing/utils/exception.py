import functools

from mlflow.exceptions import MlflowTracingException


def raise_as_trace_exception(f):
    """
    A decorator to make sure that the decorated function only raises MlflowTracingException.

    Any exceptions are caught and translated to MlflowTracingException before exiting the function.
    This is helpful for upstream functions to handle tracing related exceptions properly.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise MlflowTracingException(e) from e

    return wrapper
