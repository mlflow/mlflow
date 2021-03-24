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
from tests.autologging.fixtures import TestStream, test_mode_off, patch_destination
from tests.autologging.fixtures import reset_stderr  # pylint: disable=unused-import


pytestmark = pytest.mark.large


@pytest.fixture
def logger():
    return logging.getLogger(mlflow.__name__)


@pytest.fixture
def autolog_function(patch_destination, logger):
    def original_impl():
        # Sleep during the original function implementation to increase the likelihood of
        # overlapping session stages (i.e. simultaneous preamble / postamble / original function
        # execution states across autologging sessions) during multithreaded execution
        time.sleep(0.01)
        warnings.warn("Test warning from OG function", category=UserWarning)

    patch_destination.fn = original_impl

    def patch_impl(original):
        eprint("patch1")
        logger.info("patch2")
        warnings.warn_explicit(
            "preamble MLflow warning", category=Warning, filename=mlflow.__file__, lineno=5
        )
        warnings.warn_explicit(
            "preamble numpy warning", category=UserWarning, filename=np.__file__, lineno=7
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

    return test_autolog


def test_autologging_warnings_are_redirected_as_expected(
    autolog_function, patch_destination, logger
):
    stream = TestStream()
    sys.stderr = stream

    with pytest.warns(None) as warnings_record:
        autolog_function(silent=False)
        patch_destination.fn()

    # The following types of warnings are rerouted to MLflow's event loggers:
    # 1. All MLflow warnings emitted during patch function execution
    # 2. All warnings emitted during the patch function preamble (before the execution of the
    #    original / underlying function) and postamble (after the execution of the underlying
    #    function)
    # 3. non-MLflow warnings emitted during autologging setup / enablement
    #
    # Accordingly, we expect the following warnings to have been emitted normally: 1. MLflow
    # warnings emitted during autologging enablement, 2. non-MLflow warnings emitted during original
    # / underlying function execution
    warning_messages = set([str(w.message) for w in warnings_record])
    assert warning_messages == set(["enablement warning MLflow", "Test warning from OG function"])

    # Further, We expect MLflow's logging stream to contain content from all warnings emitted during
    # the autologging preamble and postamble and non-MLflow warnings emitted during autologging
    # enablement
    for item in [
        'MLflow autologging encountered a warning: "%s:5: Warning: preamble MLflow warning"',
        'MLflow autologging encountered a warning: "%s:10: Warning: postamble MLflow warning"',
    ]:
        assert (item % mlflow.__file__) in stream.content
    for item in [
        'MLflow autologging encountered a warning: "%s:7: UserWarning: preamble numpy warning"',
        'MLflow autologging encountered a warning: "%s:14: Warning: postamble numpy warning"',
        'MLflow autologging encountered a warning: "%s:30: Warning: enablement warning numpy"',
    ]:
        assert (item % np.__file__) in stream.content


def test_autologging_event_logging_and_warnings_respect_silent_mode(
    autolog_function, patch_destination, logger
):
    og_showwarning = warnings.showwarning
    stream = TestStream()
    sys.stderr = stream

    with pytest.warns(None) as silent_warnings_record:
        autolog_function(silent=True)
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
        autolog_function(silent=False)
        patch_destination.fn()

    # Verify that calling the autolog function with `silent=False` and invoking the mock training
    # function with autolog disabled produces event logs and warnings
    for item in ["enablement1", "enablement2", "enablement3", "enablement4"]:
        assert item in stream.content

    for item in ["patch1", "patch2", "patch3", "patch4"]:
        assert item in stream.content

    warning_messages = set([str(w.message) for w in noisy_warnings_record])
    assert "enablement warning MLflow" in warning_messages

    # Verify that `warnings.showwarning` was restored to its original value after training
    # and that MLflow event logs are enabled
    assert warnings.showwarning == og_showwarning
    logger.info("verify that event logs are enabled")
    assert "verify that event logs are enabled" in stream.content


def test_silent_mode_is_respected_in_multithreaded_environments(
    autolog_function, patch_destination, logger
):
    og_showwarning = warnings.showwarning
    stream = TestStream()
    sys.stderr = stream

    autolog_function(silent=True)

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

    with pytest.warns(None):
        with ThreadPoolExecutor(max_workers=50) as executor:
            for _ in range(100):
                executor.submit(parallel_fn)

    assert warnings.showwarning == og_showwarning
    logger.info("verify that event logs are enabled")
    assert "verify that event logs are enabled" in stream.content
