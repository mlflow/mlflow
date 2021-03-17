# pylint: disable=unused-argument

import logging
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import mlflow
from mlflow.utils.logging_utils import eprint
from mlflow.utils.autologging import autologging_integration, safe_patch

import pytest
import numpy as np
from tests.autologging.fixtures import (
    TestStream,
    test_mode_off,
    patch_destination,
)  # pylint: disable=unused-import


pytestmark = pytest.mark.large


def test_silent_mode_single_threaded(patch_destination):
    og_showwarning = warnings.showwarning
    stream = TestStream()
    sys.stderr = stream
    logger = logging.getLogger(mlflow.__name__)

    def original_impl():
        warnings.warn("Test warning from OG function", category=UserWarning)

    patch_destination.fn = original_impl

    def patch_impl(original):
        eprint("patch1")
        logger.info("patch2")
        warnings.warn_explicit(
            "preamble MLflow warning", category=Warning, filename=mlflow.__file__, lineno=5
        )
        warnings.warn_explicit(
            "preamble numpy warning", category=Warning, filename=np.__file__, lineno=7
        )
        original()
        warnings.warn_explicit(
            "postamble MLflow warning", category=Warning, filename=mlflow.__file__, lineno=10
        )
        warnings.warn_explicit(
            "postamble numpy warning", category=Warning, filename=np.__file__, lineno=14
        )
        logger.warning("patch3")
        logger.critical("patch4")

    @autologging_integration("test_integration")
    def test_autolog(disable=False, silent=False):
        eprint("enablement1")
        logger.info("enablement2")
        logger.warning("enablement3")
        logger.critical("enablement4")
        warnings.warn_explicit(
            "enablement warning MLflow", category=Warning, filename=mlflow.__file__, lineno=15
        )
        warnings.warn_explicit(
            "enablement warning numpy", category=Warning, filename=np.__file__, lineno=30
        )
        safe_patch("test_integration", patch_destination, "fn", patch_impl)

    with pytest.warns(None) as silent_warnings_record:
        test_autolog(silent=True)
        patch_destination.fn()

    assert len(silent_warnings_record) == 1
    assert "Test warning from OG function" in str(silent_warnings_record[0].message)
    assert stream.content is None

    # Verify that `warnings.showwarning` was restored to its original value after training
    # and that MLflow event logs are enabled
    assert warnings.showwarning == og_showwarning
    logger.info("verify that event logs are enabled")
    assert "verify that event logs are enabled" in stream.content

    stream.reset()

    with pytest.warns(None) as noisy_warnings_record:
        test_autolog(silent=False)
        patch_destination.fn()

    # MLflow warnings emitted during patch function execution are rerouted to MLflow's event
    # loggers. Accordingly, we expect the following warnings to have been captured:
    # 1. MLflow & non-MLflow warnings emitted during autologging enablement, 2. non-MLflow warnings
    # emitted during patch code execution, and 3. non-MLflow warnings emitted during original
    # / underlying function execution
    assert len(noisy_warnings_record) == 5
    warning_messages = [str(w.message) for w in noisy_warnings_record]
    assert "enablement warning MLflow" in warning_messages
    assert "enablement warning numpy" in warning_messages
    assert "preamble numpy warning" in warning_messages
    assert "postamble numpy warning" in warning_messages
    assert "Test warning from OG function" in warning_messages

    for item in ["enablement1", "enablement2", "enablement3", "enablement4"]:
        assert item in stream.content

    for item in ["patch1", "patch2", "patch3", "patch4"]:
        assert item in stream.content

    # MLflow warnings emitted during patch function execution are rerouted to MLflow's event
    # loggers. Accordingly, we expect MLflow's logging stream to contain content from warnings
    # emitted during the autologging preamble and postamble
    for item in ["preamble MLflow warning", "postamble MLflow warning"]:
        assert item in stream.content

    # Verify that `warnings.showwarning` was restored to its original value after training
    # and that MLflow event logs are enabled
    assert warnings.showwarning == og_showwarning
    logger.info("verify that event logs are enabled")
    assert "verify that event logs are enabled" in stream.content


def test_silent_mode_multithreaded(patch_destination):
    og_showwarning = warnings.showwarning
    stream = TestStream()
    sys.stderr = stream
    logger = logging.getLogger(mlflow.__name__)

    def original_impl():
        # Sleep during the original function implementation to increase the likelihood of
        # overlapping session stages (i.e. simultaneous preamble / postamble / original function
        # execution states across autologging sessions)
        time.sleep(0.01)
        warnings.warn("Test warning from OG function", category=UserWarning)

    patch_destination.fn = original_impl

    def patch_impl(original):
        logger.info("preamble event")
        warnings.warn_explicit(
            "preamble warning", category=Warning, filename=mlflow.__file__, lineno=5
        )
        original()
        logger.info("postamble event")
        logger.info("patch preamble")
        warnings.warn_explicit(
            "postamble warning", category=Warning, filename=np.__file__, lineno=10
        )

    @autologging_integration("test_integration")
    def test_autolog(disable=False, silent=False):
        logger.warning("enablement")
        warnings.warn_explicit(
            "enablement warning", category=Warning, filename=mlflow.__file__, lineno=15
        )
        safe_patch("test_integration", patch_destination, "fn", patch_impl)

    test_autolog(silent=True)

    def parallel_fn():
        # Sleep for a random interval to increase the likelihood of overlapping session stages
        # (i.e. simultaneous preamble / postamble / original function execution states across
        # autologging sessions)
        time.sleep(np.random.random())
        patch_destination.fn()
        return True

    executions = []
    with pytest.warns(None) as warnings_record:
        with ThreadPoolExecutor(max_workers=50) as executor:
            for _ in range(100):
                executions.append(executor.submit(parallel_fn))

    assert all([e.result() is True for e in executions])

    # Verify that all warnings and log events from MLflow autologging code were silenced
    # and that all warnings from the original / underlying routine were emitted as normal
    assert stream.content is None
    assert len(warnings_record) == 100
    assert all(["Test warning from OG function" in str(w.message) for w in warnings_record])

    # Verify that `warnings.showwarning` was restored to its original value after training
    # and that MLflow event logs are enabled
    assert warnings.showwarning == og_showwarning
    logger.info("verify that event logs are enabled")
    assert "verify that event logs are enabled" in stream.content


@pytest.mark.usefixtures(test_mode_off.__name__)
def test_silent_mode_restores_warning_and_event_logging_behavior_correctly_if_errors_occur():
    og_showwarning = warnings.showwarning
    stream = TestStream()
    sys.stderr = stream
    logger = logging.getLogger(mlflow.__name__)

    def original_impl():
        # Sleep during the original function implementation to increase the likelihood of
        # overlapping session stages (i.e. simultaneous preamble / postamble / original function
        # execution states across autologging sessions)
        raise Exception("original error")

    patch_destination.fn = original_impl

    def patch_impl(original):
        raise Exception("preamble error")
        original()
        raise Exception("postamble error")

    @autologging_integration("test_integration")
    def test_autolog(disable=False, silent=False):
        safe_patch("test_integration", patch_destination, "fn", patch_impl)
        raise Exception("enablement error")

    def parallel_fn():
        # Sleep for a random interval to increase the likelihood of overlapping session stages
        # (i.e. simultaneous preamble / postamble / original function execution states across
        # autologging sessions)
        time.sleep(np.random.random())
        patch_destination.fn()

    with pytest.raises(Exception):
        test_autolog(silent=True)

    with pytest.warns(None) as warnings_record:
        with ThreadPoolExecutor(max_workers=50) as executor:
            for _ in range(100):
                executor.submit(parallel_fn)

    assert warnings.showwarning == og_showwarning
    logger.info("verify that event logs are enabled")
    assert "verify that event logs are enabled" in stream.content
