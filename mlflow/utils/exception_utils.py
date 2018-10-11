import traceback
import sys

from six import reraise

def _log_exception_trace_and_reraise(reraised_error_type, reraised_error_text, **exception_kwargs):
    """
    Logs information about an exception that is currently being handled
    and reraises it with the specified error text as a message.

    :param reraised_error_type: The class of exception to reraise
    :param reraised_error_text: The text of the reraised exception to include with the current
                                exception's traceback.
    :param exception_kwargs: Keyword arguments to include in the exception constructor
    """
    traceback.print_exc()
    tb = sys.exc_info()[2]
    reraise(reraised_error_type, reraised_error_type(reraised_error_text, **exception_kwargs), tb)
