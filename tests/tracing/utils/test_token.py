import logging

from mlflow.tracing.utils.token import suppress_token_detach_warning_to_debug_level


def test_suppress_token_detach_warning(caplog):
    logger = logging.getLogger("opentelemetry.context")

    logger.exception("Failed to detach context")
    assert caplog.records[0].message == "Failed to detach context"
    assert caplog.records[0].levelname == "ERROR"

    suppress_token_detach_warning_to_debug_level()

    logger.exception("Failed to detach context")  # This should be logged as DEBUG level
    logger.exception("Another error")  # This should be logged as is

    assert caplog.records[1].message == "Failed to detach context"
    assert caplog.records[1].levelname == "DEBUG"

    assert caplog.records[2].message == "Another error"
    assert caplog.records[2].levelname == "ERROR"
