import logging

from mlflow.tracing.utils.token import suppress_token_detach_warning_to_debug_level


def test_suppress_token_detach_warning(caplog):
    logger = logging.getLogger("opentelemetry.context")
    logger.setLevel(logging.INFO)
    logger.removeFilter(logger.filters[0])

    logger.exception("Failed to detach context")
    assert caplog.records[0].message == "Failed to detach context"
    assert caplog.records[0].levelname == "ERROR"

    suppress_token_detach_warning_to_debug_level()

    logger.exception("Failed to detach context")
    assert len(caplog.records) == 1  # If the log level is not debug, the log shouldn't be recorded

    logger.exception("Another error")  # Other type of error log should still be recorded
    assert len(caplog.records) == 2
    assert caplog.records[1].message == "Another error"

    # If we change the log level to debug, the log should be recorded at the debug level
    logger.setLevel(logging.DEBUG)

    logger.exception("Failed to detach context")

    assert caplog.records[2].message == "Failed to detach context"
    assert caplog.records[2].levelname == "DEBUG"
