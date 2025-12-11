import contextlib
import logging
import logging.config
import re
import sys

from mlflow.environment_variables import MLFLOW_LOGGING_LEVEL
from mlflow.utils.thread_utils import ThreadLocalVariable

# Logging format example:
# 2018/11/20 12:36:37 INFO mlflow.sagemaker: Creating new SageMaker endpoint
LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"


class MlflowLoggingStream:
    """
    A Python stream for use with event logging APIs throughout MLflow (`eprint()`,
    `logger.info()`, etc.). This stream wraps `sys.stderr`, forwarding `write()` and
    `flush()` calls to the stream referred to by `sys.stderr` at the time of the call.
    It also provides capabilities for disabling the stream to silence event logs.
    """

    def __init__(self):
        self._enabled = True

    def write(self, text):
        if self._enabled:
            sys.stderr.write(text)

    def flush(self):
        if self._enabled:
            sys.stderr.flush()

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value


MLFLOW_LOGGING_STREAM = MlflowLoggingStream()


def disable_logging():
    """
    Disables the `MlflowLoggingStream` used by event logging APIs throughout MLflow
    (`eprint()`, `logger.info()`, etc), silencing all subsequent event logs.
    """
    MLFLOW_LOGGING_STREAM.enabled = False


def enable_logging():
    """
    Enables the `MlflowLoggingStream` used by event logging APIs throughout MLflow
    (`eprint()`, `logger.info()`, etc), emitting all subsequent event logs. This
    reverses the effects of `disable_logging()`.
    """
    MLFLOW_LOGGING_STREAM.enabled = True


class MlflowFormatter(logging.Formatter):
    """
    Custom Formatter Class to support colored log
    ANSI characters might not work natively on older Windows, so disabling the feature for win32.
    See https://github.com/borntyping/python-colorlog/blob/dfa10f59186d3d716aec4165ee79e58f2265c0eb/colorlog/escape_codes.py#L16C8-L16C31
    """

    # Copied from color log package https://github.com/borntyping/python-colorlog/blob/dfa10f59186d3d716aec4165ee79e58f2265c0eb/colorlog/escape_codes.py#L33-L50
    COLORS = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "purple": 35,
        "cyan": 36,
        "white": 37,
        "light_black": 90,
        "light_red": 91,
        "light_green": 92,
        "light_yellow": 93,
        "light_blue": 94,
        "light_purple": 95,
        "light_cyan": 96,
        "light_white": 97,
    }
    RESET = "\033[0m"

    def format(self, record):
        if color := getattr(record, "color", None):
            if color in self.COLORS and sys.platform != "win32":
                color_code = self._escape(self.COLORS[color])
                return f"{color_code}{super().format(record)}{self.RESET}"
        return super().format(record)

    def _escape(self, code: int) -> str:
        return f"\033[{code}m"


# Thread-local variable to suppress logs in the certain thread, used
# in telemetry client to suppress logs in the consumer thread
should_suppress_logs_in_thread = ThreadLocalVariable(default_factory=lambda: False)


class SuppressLogFilter(logging.Filter):
    def filter(self, record):
        if should_suppress_logs_in_thread.get():
            return False
        return super().filter(record)


def _configure_mlflow_loggers(root_module_name):
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "mlflow_formatter": {
                    "()": MlflowFormatter,
                    "format": LOGGING_LINE_FORMAT,
                    "datefmt": LOGGING_DATETIME_FORMAT,
                },
            },
            "handlers": {
                "mlflow_handler": {
                    "formatter": "mlflow_formatter",
                    "class": "logging.StreamHandler",
                    "stream": MLFLOW_LOGGING_STREAM,
                    "filters": ["suppress_in_thread"],
                },
            },
            "loggers": {
                root_module_name: {
                    "handlers": ["mlflow_handler"],
                    "level": (MLFLOW_LOGGING_LEVEL.get() or "INFO").upper(),
                    "propagate": False,
                },
                "sqlalchemy.engine": {
                    "handlers": ["mlflow_handler"],
                    "level": "WARN",
                    "propagate": False,
                },
                "alembic": {
                    "handlers": ["mlflow_handler"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
            "filters": {
                "suppress_in_thread": {
                    "()": SuppressLogFilter,
                }
            },
        }
    )


def eprint(*args, **kwargs):
    print(*args, file=MLFLOW_LOGGING_STREAM, **kwargs)


class LoggerMessageFilter(logging.Filter):
    def __init__(self, module: str, filter_regex: re.Pattern):
        super().__init__()
        self._pattern = filter_regex
        self._module = module

    def filter(self, record):
        if record.name == self._module and self._pattern.search(record.msg):
            return False
        return True


@contextlib.contextmanager
def suppress_logs(module: str, filter_regex: re.Pattern):
    """
    Context manager that suppresses log messages from the specified module that match the specified
    regular expression. This is useful for suppressing expected log messages from third-party
    libraries that are not relevant to the current test.
    """
    logger = logging.getLogger(module)
    filter = LoggerMessageFilter(module=module, filter_regex=filter_regex)
    logger.addFilter(filter)
    try:
        yield
    finally:
        logger.removeFilter(filter)


def _debug(s: str) -> None:
    """
    Debug function to test logging level.
    """
    logging.getLogger(__name__).debug(s)


@contextlib.contextmanager
def suppress_logs_in_thread():
    """
    Context manager to suppress logs in the current thread.
    """
    original_value = should_suppress_logs_in_thread.get()
    try:
        should_suppress_logs_in_thread.set(True)
        yield
    finally:
        should_suppress_logs_in_thread.set(original_value)
