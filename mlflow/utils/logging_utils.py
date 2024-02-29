import contextlib
import logging
import logging.config
import re
import sys

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


def _configure_mlflow_loggers(root_module_name):
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "mlflow_formatter": {
                    "format": LOGGING_LINE_FORMAT,
                    "datefmt": LOGGING_DATETIME_FORMAT,
                },
            },
            "handlers": {
                "mlflow_handler": {
                    "formatter": "mlflow_formatter",
                    "class": "logging.StreamHandler",
                    "stream": MLFLOW_LOGGING_STREAM,
                },
            },
            "loggers": {
                root_module_name: {
                    "handlers": ["mlflow_handler"],
                    "level": "INFO",
                    "propagate": False,
                },
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
