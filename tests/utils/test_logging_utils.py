import logging
import os
import re
import subprocess
import sys
import uuid
from io import StringIO

import pytest

import mlflow
from mlflow.utils import logging_utils
from mlflow.utils.logging_utils import LOGGING_LINE_FORMAT, eprint, suppress_logs

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
    with suppress_logs(module, re.compile(r"This .* be suppressed.")):
        logger.error(message)
    assert len(capture_stream.getvalue()) == 0


@pytest.mark.parametrize(
    ("log_level", "expected"),
    [
        ("DEBUG", True),
        ("INFO", False),
        ("NOTSET", False),
    ],
)
def test_logging_level(log_level: str, expected: bool) -> None:
    random_str = str(uuid.uuid4())
    stdout = subprocess.check_output(
        [
            sys.executable,
            "-c",
            f"from mlflow.utils.logging_utils import _debug; _debug({random_str!r})",
        ],
        env=os.environ.copy() | {"MLFLOW_LOGGING_LEVEL": log_level},
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert (random_str in stdout) is expected


@pytest.mark.parametrize(
    "env_var_name",
    ["MLFLOW_CONFIGURE_LOGGING", "MLFLOW_LOGGING_CONFIGURE_LOGGING"],
)
@pytest.mark.parametrize(
    "value",
    ["0", "1"],
)
def test_mlflow_configure_logging_env_var(env_var_name: str, value: str) -> None:
    expected_level = logging.INFO if value == "1" else logging.WARNING
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            f"""
import logging
import mlflow

assert logging.getLogger("mlflow").isEnabledFor({expected_level})
""",
        ],
        env=os.environ.copy() | {env_var_name: value},
    )


@pytest.mark.parametrize("configure_logging", ["0", "1"])
def test_alembic_logging_respects_configure_flag(configure_logging: str, tmp_sqlite_uri: str):
    user_specified_format = "CUSTOM: %(name)s - %(message)s"
    actual_format = user_specified_format if configure_logging == "0" else LOGGING_LINE_FORMAT
    code = f"""
import logging

# user-specified format, this should only take effect if configure_logging is 0
logging.basicConfig(level=logging.INFO, format={user_specified_format!r})

import mlflow

# Check the alembic logger format, which is now configured in _configure_mlflow_loggers
alembic_logger = logging.getLogger("alembic")
if {configure_logging!r} == "1":
    # When MLFLOW_CONFIGURE_LOGGING is enabled, alembic logger has its own handler
    assert len(alembic_logger.handlers) > 0
    actual_format = alembic_logger.handlers[0].formatter._fmt
else:
    # When MLFLOW_CONFIGURE_LOGGING is disabled, alembic logger propagates to root
    assert alembic_logger.propagate
    root_logger = logging.getLogger()
    actual_format = root_logger.handlers[0].formatter._fmt

assert actual_format == {actual_format!r}, actual_format
"""
    subprocess.check_call(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "MLFLOW_TRACKING_URI": tmp_sqlite_uri,
            "MLFLOW_CONFIGURE_LOGGING": configure_logging,
        },
    )
