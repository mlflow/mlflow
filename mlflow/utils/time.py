import datetime
import time

from pytz import reference


def get_current_time_millis():
    """
    Returns the time in milliseconds since the epoch as an integer number.
    """
    return int(time.time() * 1000)


def conv_longdate_to_str(longdate, local_tz=True):
    date_time = datetime.datetime.fromtimestamp(longdate / 1000.0)
    str_long_date = date_time.strftime("%Y-%m-%d %H:%M:%S")
    if local_tz:
        str_long_date += " " + reference.LocalTimezone().tzname(date_time)

    return str_long_date


class Timer:
    """
    Measures elapsed time.

    .. code-block:: python

        from mlflow.utils.time import Timer

        with Timer() as t:
            ...

        print(f"Elapsed time: {t:.2f} seconds")
    """

    def __init__(self):
        self.time = 0.0

    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.time = time.perf_counter() - self.time

    def __format__(self, format_spec: str) -> str:
        return self.time.__format__(format_spec)

    def __repr__(self) -> str:
        return self.time.__repr__()

    def __str__(self) -> str:
        return self.time.__str__()
