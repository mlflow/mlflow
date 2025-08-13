import importlib
import logging

_logger = logging.getLogger(__name__)


class LogDemotionFilter(logging.Filter):
    def __init__(self, module: str, message: str):
        super().__init__()
        self.module = module
        self.message = message

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == self.module and self.message in record.getMessage():
            record.levelno = logging.DEBUG  # Change the log level to DEBUG
            record.levelname = "DEBUG"

            # Check the log level for the logger is debug or not
            logger = logging.getLogger(self.module)
            return logger.isEnabledFor(logging.DEBUG)
        return True

    def __eq__(self, other):
        if isinstance(other, LogDemotionFilter):
            return self.module == other.module and self.message == other.message
        return False


def suppress_warning(module: str, message: str):
    """
    Convert the "Failed to detach context" log raised by the OpenTelemetry logger to DEBUG
    level so that it does not show up in the user's console.

    Args:
        module: The module name of the logger that raises the warning.
        message: The (part of) message in the log that needs to be demoted to DEBUG level
    """
    try:
        logger = getattr(importlib.import_module(module), "logger", None)
        log_filter = LogDemotionFilter(module, message)
        if logger and not any(f == log_filter for f in logger.filters):
            logger.addFilter(log_filter)
    except Exception as e:
        _logger.debug(f"Failed to suppress the warning for {module}", exc_info=e)
        raise
