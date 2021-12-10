# pylint: disable=unused-argument

import logging
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import mlflow
from mlflow.utils.logging_utils import eprint
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

import pytest
import numpy as np
from tests.autologging.fixtures import test_mode_off, patch_destination
from tests.autologging.fixtures import reset_stderr  # pylint: disable=unused-import


pytestmark = pytest.mark.large


@pytest.fixture
def logger():
    return logging.getLogger(mlflow.__name__)


@pytest.fixture
def autolog_function(patch_destination, logger):
    def original_impl():
        # Increase the duration of the original function by inserting a short sleep in order to
        # increase the likelihood of overlapping session stages (i.e. simultaneous preamble /
        # postamble / original function execution states across autologging sessions) during
        # multithreaded execution. We use a duration of 50 milliseconds to avoid slowing down the
        # test significantly
        time.sleep(0.05)
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
    stream = StringIO()
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
        assert (item % mlflow.__file__) in stream.getvalue()
    for item in [
        'MLflow autologging encountered a warning: "%s:7: UserWarning: preamble numpy warning"',
        'MLflow autologging encountered a warning: "%s:14: Warning: postamble numpy warning"',
        'MLflow autologging encountered a warning: "%s:30: Warning: enablement warning numpy"',
    ]:
        assert (item % np.__file__) in stream.getvalue()


def test_autologging_event_logging_and_warnings_respect_silent_mode(
    autolog_function, patch_destination, logger
):
    og_showwarning = warnings.showwarning
    stream = StringIO()
    sys.stderr = stream

    with pytest.warns(None) as silent_warnings_record:
        autolog_function(silent=True)
        patch_destination.fn()

    assert len(silent_warnings_record) == 1
    assert "Test warning from OG function" in str(silent_warnings_record[0].message)
    assert not stream.getvalue()

    # Verify that `warnings.showwarning` was restored to its original value after training
    # and that MLflow event logs are enabled
    assert warnings.showwarning == og_showwarning
    logger.info("verify that event logs are enabled")
    assert "verify that event logs are enabled" in stream.getvalue()

    stream.truncate(0)

    with pytest.warns(None) as noisy_warnings_record:
        autolog_function(silent=False)
        patch_destination.fn()

    # Verify that calling the autolog function with `silent=False` and invoking the mock training
    # function with autolog disabled produces event logs and warnings
    for item in ["enablement1", "enablement2", "enablement3", "enablement4"]:
        assert item in stream.getvalue()

    for item in ["patch1", "patch2", "patch3", "patch4"]:
        assert item in stream.getvalue()

    warning_messages = set([str(w.message) for w in noisy_warnings_record])
    assert "enablement warning MLflow" in warning_messages

    # Verify that `warnings.showwarning` was restored to its original value after training
    # and that MLflow event logs are enabled
    assert warnings.showwarning == og_showwarning
    logger.info("verify that event logs are enabled")
    assert "verify that event logs are enabled" in stream.getvalue()


def test_silent_mode_is_respected_in_multithreaded_environments(
    autolog_function, patch_destination, logger
):
    og_showwarning = warnings.showwarning
    stream = StringIO()
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
    assert not stream.getvalue()
    assert len(warnings_record) == 100
    assert all(["Test warning from OG function" in str(w.message) for w in warnings_record])

    # Verify that `warnings.showwarning` was restored to its original value after training
    # and that MLflow event logs are enabled
    assert warnings.showwarning == og_showwarning
    logger.info("verify that event logs are enabled")
    assert "verify that event logs are enabled" in stream.getvalue()


@pytest.mark.usefixtures(test_mode_off.__name__)
def test_silent_mode_restores_warning_and_event_logging_behavior_correctly_if_errors_occur():
    og_showwarning = warnings.showwarning
    stream = StringIO()
    sys.stderr = stream
    logger = logging.getLogger(mlflow.__name__)

    def original_impl():
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
        # Sleep for a random duration between 0 and 1 seconds to increase the likelihood of
        # overlapping session stages (i.e. simultaneous preamble / postamble / original function
        # execution states across autologging sessions)
        time.sleep(np.random.random())
        patch_destination.fn()

    with pytest.raises(Exception, match="enablement error"):
        test_autolog(silent=True)

    with pytest.warns(None):
        with ThreadPoolExecutor(max_workers=50) as executor:
            for _ in range(100):
                executor.submit(parallel_fn)

    assert warnings.showwarning == og_showwarning
    logger.info("verify that event logs are enabled")
    assert "verify that event logs are enabled" in stream.getvalue()


def test_silent_mode_operates_independently_across_integrations(patch_destination, logger):
    stream = StringIO()
    sys.stderr = stream

    patch_destination.fn2 = lambda *args, **kwargs: "fn2"

    def patch_impl1(original):
        warnings.warn("patchimpl1")
        original()

    @autologging_integration("integration1")
    def autolog1(disable=False, silent=False):
        logger.info("autolog1")
        safe_patch("integration1", patch_destination, "fn", patch_impl1)

    def patch_impl2(original):
        logger.info("patchimpl2")
        original()

    @autologging_integration("integration2")
    def autolog2(disable=False, silent=False):
        warnings.warn_explicit(
            "warn_autolog2", category=Warning, filename=mlflow.__file__, lineno=5
        )
        logger.info("event_autolog2")
        safe_patch("integration2", patch_destination, "fn2", patch_impl2)

    with pytest.warns(None) as warnings_record:
        autolog1(silent=True)
        autolog2(silent=False)

        patch_destination.fn()
        patch_destination.fn2()

    warning_messages = [str(w.message) for w in warnings_record]
    assert warning_messages == ["warn_autolog2"]

    assert "autolog1" not in stream.getvalue()
    assert "patchimpl1" not in stream.getvalue()

    assert "event_autolog2" in stream.getvalue()
    assert "patchimpl2" in stream.getvalue()


@pytest.mark.parametrize("silent", [False, True])
@pytest.mark.parametrize("disable", [False, True])
def test_silent_mode_and_warning_rerouting_respect_disabled_flag(
    patch_destination, silent, disable
):
    stream = StringIO()
    sys.stderr = stream

    def original_fn():
        warnings.warn("Test warning", category=UserWarning)

    patch_destination.fn = original_fn

    @autologging_integration("test_integration")
    def test_autolog(disable=False, silent=False):
        safe_patch("test_integration", patch_destination, "fn", lambda original: original())

    test_autolog(disable=disable, silent=silent)

    with warnings.catch_warnings(record=True) as warnings_record:
        patch_destination.fn()

    # Verify that calling the patched instance method still emits the expected warning
    assert len(warnings_record) == 1
    assert warnings_record[0].message.args[0] == "Test warning"
    assert warnings_record[0].category == UserWarning

    # Verify that nothing is printed to the stderr-backed MLflow event logger, which would indicate
    # rerouting of warning content
    assert not stream.getvalue()
