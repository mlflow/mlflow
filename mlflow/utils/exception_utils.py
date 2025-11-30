import traceback


def get_stacktrace(error):
    msg = repr(error)
    try:
        tb = traceback.format_exception(error)
        return (msg + "".join(tb)).strip()
    except Exception:
        return msg
