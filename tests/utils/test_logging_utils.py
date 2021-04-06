import logging
import sys
import pytest

import mlflow
import mlflow.utils.logging_utils as logging_utils
from mlflow.utils.logging_utils import eprint

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


class TestStream:
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
    stream1 = TestStream()
    stream2 = TestStream()
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
    stream = TestStream()
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
    stream = TestStream()
    sys.stderr = stream

    eprint("foo", flush=True)
    assert "foo" in stream.content
    assert stream.flush_count > 0
