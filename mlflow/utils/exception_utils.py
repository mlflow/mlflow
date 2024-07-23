import sys
import traceback


def get_stacktrace(error):
    msg = repr(error)
    try:
        if sys.version_info < (3, 10):
            tb = traceback.format_exception(error.__class__, error, error.__traceback__)
        else:
            tb = traceback.format_exception(error)
        return (msg + "\n\n".join(tb)).strip()
    except Exception:
        return msg
