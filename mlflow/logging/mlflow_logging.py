import threading
import logging as _logging
import sys as _sys

from logging import INFO

# Don't use this directly. Use _get_logger() instead.
_logger = None
_logger_lock = threading.Lock()


def _get_logger():
    global _logger

    # Use double-checked locking to avoid taking lock unnecessarily.
    if _logger:
        return _logger

    _logger_lock.acquire()

    try:
        if _logger:
            return _logger

        # Scope the TensorFlow logger to not conflict with users' loggers.
        logger = _logging.getLogger('tensorflow')

        # Don't further configure the MLflow logger if the root logger is
        # already configured. This prevents double logging in those cases.
        if not _logging.getLogger().handlers:
            # Determine whether we are in an interactive environment
            _interactive = False
            try:
                # This is only defined in interactive shells.
                if _sys.ps1:
                    _interactive = True
            except AttributeError:
                # Even now, we may be in an interactive shell with `python -i`.
                _interactive = _sys.flags.interactive

            # If we are in an interactive environment (like Jupyter), set loglevel
            # to INFO and pipe the output to stdout.
            if _interactive:
                logger.setLevel(INFO)
                _logging_target = _sys.stdout
            else:
                _logging_target = _sys.stderr

            # Add the output handler.
            _handler = _logging.StreamHandler(_logging_target)
            _handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
            logger.addHandler(_handler)

        _logger = logger
        return _logger

    finally:
        _logger_lock.release()


def debug(msg, *args, **kwargs):
    _get_logger().debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    _get_logger().error(msg, *args, **kwargs)


def fatal(msg, *args, **kwargs):
    _get_logger().fatal(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    _get_logger().info(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    _get_logger().warn(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    _get_logger().warning(msg, *args, **kwargs)


def set_verbosity(verbosity):
    _get_logger().setLevel(verbosity)
