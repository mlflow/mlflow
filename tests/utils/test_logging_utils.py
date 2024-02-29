import logging
import re
import sys
from io import StringIO

import pytest

import mlflow
from mlflow.utils import logging_utils
from mlflow.utils.logging_utils import eprint, suppress_logs

logger = logging.getLogger(mlflow.__name__)

LOGGING_FNS_TO_TEST = [logger.info, logger.warning, logger.critical, eprint]


@pytest.fixture(autouse=True)
def reset_stderr():
    prev_stderr = sys.stderr
    yield
    sys.stderr = prev_stderr


@pytest.fixture(autouse=True)
def reset_logging_enablement():
    yield
    logging_utils.enable_logging()


@pytest.fixture(autouse=True)
def reset_logging_level():
    level_before = logger.level
    yield
    logger.setLevel(level_before)


class SampleStream:
    def __init__(self):
        self.content = None
        self.flush_count = 0

    def write(self, text):
        self.content = (self.content or "") + text

    def flush(self):
        self.flush_count += 1

    def reset(self):
        self.content = None
        self.flush_count = 0


@pytest.mark.parametrize("logging_fn", LOGGING_FNS_TO_TEST)
def test_event_logging_apis_respect_stderr_reassignment(logging_fn):
    stream1 = SampleStream()
    stream2 = SampleStream()
    message_content = "test message"

    sys.stderr = stream1
    assert stream1.content is None
    logging_fn(message_content)
    assert message_content in stream1.content
    assert stream2.content is None
    stream1.reset()

    sys.stderr = stream2
    assert stream2.content is None
    logging_fn(message_content)
    assert message_content in stream2.content
    assert stream1.content is None


@pytest.mark.parametrize("logging_fn", LOGGING_FNS_TO_TEST)
def test_event_logging_apis_respect_stream_disablement_enablement(logging_fn):
    stream = SampleStream()
    sys.stderr = stream
    message_content = "test message"

    assert stream.content is None
    logging_fn(message_content)
    assert message_content in stream.content
    stream.reset()

    logging_utils.disable_logging()
    logging_fn(message_content)
    assert stream.content is None
    stream.reset()

    logging_utils.enable_logging()
    assert stream.content is None
    logging_fn(message_content)
    assert message_content in stream.content


def test_event_logging_stream_flushes_properly():
    stream = SampleStream()
    sys.stderr = stream

    eprint("foo", flush=True)
    assert "foo" in stream.content
    assert stream.flush_count > 0


def test_debug_logs_emitted_correctly_when_configured():
    stream = SampleStream()
    sys.stderr = stream

    logger.setLevel(logging.DEBUG)
    logger.debug("test debug")
    assert "test debug" in stream.content


def test_suppress_logs():
    module = "test_logger"
    logger = logging.getLogger(module)

    message = "This message should be suppressed."

    capture_stream = StringIO()
    stream_handler = logging.StreamHandler(capture_stream)
    logger.addHandler(stream_handler)

    logger.error(message)
    assert message in capture_stream.getvalue()

    capture_stream.truncate(0)
    with suppress_logs(module, re.compile("This .* be suppressed.")):
        logger.error(message)
    assert len(capture_stream.getvalue()) == 0
