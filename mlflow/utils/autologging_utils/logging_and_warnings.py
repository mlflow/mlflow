import os
import warnings
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from threading import RLock
from threading import get_ident as get_current_thread_id

import mlflow
from mlflow.utils import logging_utils


@contextmanager
def set_warning_behavior_during_autologging(autologging_integration: str):
    """
    Reroute warnings encountered during the patch function implementation to an MLflow event
    logger, and enforce silent mode if applicable (i.e. if the corresponding autologging
    integration was called with `silent=True`), hiding MLflow event logging statements and
    hiding all warnings in the autologging preamble and postamble (i.e. the code surrounding
    the user's original / underlying ML function). Non-MLflow warnings are enabled during the
    execution of the original / underlying ML function

    Note that we've opted *not* to apply this context manager as a decorator on
    `safe_patch_function` because the context-manager-as-decorator pattern uses
    `contextlib.ContextDecorator`, which creates generator expressions that cannot be pickled
    during model serialization by ML frameworks such as scikit-learn
    """
    from mlflow.utils.autologging_utils import get_autologging_config

    is_silent_mode = get_autologging_config(autologging_integration, "silent", False)
    with (
        MlflowEventsAndWarningsBehavior(
            # MLflow warnings emitted during autologging training sessions are likely not
            # actionable and result from the autologging implementation invoking another MLflow
            # API. Accordingly, we reroute these warnings to the MLflow event logger with level
            # WARNING For reference, see recommended warning and event logging behaviors from
            # https://docs.python.org/3/howto/logging.html#when-to-use-logging
            reroute_warnings=True,
            disable_event_logs=is_silent_mode,
            disable_warnings=is_silent_mode,
        ),
        NonMlflowWarningsBehaviorForCurrentThread(
            # non-MLflow Warnings emitted during the autologging preamble (before the original /
            # underlying ML function is called) and postamble (after the original / underlying
            # ML function is called) are likely not actionable and result from the autologging
            # implementation invoking an API from a dependent library. Accordingly, we reroute
            # these warnings to the MLflow event logger with level WARNING. For reference, see
            # recommended warning and event logging behaviors from
            # https://docs.python.org/3/howto/logging.html#when-to-use-logging
            reroute_warnings=True,
            disable_warnings=is_silent_mode,
        ),
    ):
        yield


@asynccontextmanager
async def async_set_warning_behavior_during_autologging(autologging_integration: str):
    """
    Async version of `set_warning_behavior_during_autologging`
    """
    from mlflow.utils.autologging_utils import get_autologging_config

    is_silent_mode = get_autologging_config(autologging_integration, "silent", False)
    async with (
        MlflowEventsAndWarningsBehavior(
            reroute_warnings=True,
            disable_event_logs=is_silent_mode,
            disable_warnings=is_silent_mode,
        ),
        NonMlflowWarningsBehaviorForCurrentThread(
            reroute_warnings=True,
            disable_warnings=is_silent_mode,
        ),
    ):
        yield


class _WarningsController:
    """
    Provides threadsafe utilities to modify warning behavior for MLflow autologging, including:

    - Global disablement of MLflow warnings across all threads
    - Global rerouting of MLflow warnings to an MLflow event logger (i.e. `logger.warning()`)
      across all threads
    - Disablement of non-MLflow warnings for the current thread
    - Rerouting of non-MLflow warnings to an MLflow event logger for the current thread
    """

    def __init__(self):
        self._mlflow_root_path = Path(os.path.dirname(mlflow.__file__)).resolve()
        self._state_lock = RLock()

        self._did_patch_showwarning = False
        self._original_showwarning = None

        self._disabled_threads = set()
        self._rerouted_threads = set()
        self._mlflow_warnings_disabled_globally = False
        self._mlflow_warnings_rerouted_to_event_logs = False

    def _patched_showwarning(self, message, category, filename, lineno, *args, **kwargs):
        """
        A patched implementation of `warnings.showwarning` that enforces the warning configuration
        options configured on the controller (e.g. rerouting or disablement of MLflow warnings,
        disablement of all warnings for the current thread).

        Note that reassigning `warnings.showwarning` is the standard / recommended approach for
        modifying warning message display behaviors. For reference, see
        https://docs.python.org/3/library/warnings.html#warnings.showwarning
        """
        # NB: We explicitly avoid blocking on the `self._state_lock` lock during `showwarning`
        # to so that threads don't have to execute serially whenever they emit warnings with
        # `warnings.warn()`. We only lock during configuration changes to ensure that
        # `warnings.showwarning` is patched or unpatched at the correct times.

        from mlflow.utils.autologging_utils import _logger

        # If the warning's source file is contained within the MLflow package's base
        # directory, it is an MLflow warning and should be emitted via `logger.warning`
        warning_source_path = Path(filename).resolve()
        is_mlflow_warning = self._mlflow_root_path in warning_source_path.parents
        curr_thread = get_current_thread_id()

        if (curr_thread in self._disabled_threads) or (
            is_mlflow_warning and self._mlflow_warnings_disabled_globally
        ):
            return
        elif (curr_thread in self._rerouted_threads and not is_mlflow_warning) or (
            is_mlflow_warning and self._mlflow_warnings_rerouted_to_event_logs
        ):
            _logger.warning(
                'MLflow autologging encountered a warning: "%s:%d: %s: %s"',
                filename,
                lineno,
                category.__name__,
                message,
            )
        else:
            self._original_showwarning(message, category, filename, lineno, *args, **kwargs)

    def _should_patch_showwarning(self):
        return (
            (len(self._disabled_threads) > 0)
            or (len(self._rerouted_threads) > 0)
            or self._mlflow_warnings_disabled_globally
            or self._mlflow_warnings_rerouted_to_event_logs
        )

    def _modify_patch_state_if_necessary(self):
        """
        Patches or unpatches `warnings.showwarning` if necessary, as determined by:
            - Whether or not `warnings.showwarning` is already patched
            - Whether or not any custom warning state has been configured on the warnings
              controller (i.e. disablement or rerouting of certain warnings globally or for a
              particular thread)

        Note that reassigning `warnings.showwarning` is the standard / recommended approach for
        modifying warning message display behaviors. For reference, see
        https://docs.python.org/3/library/warnings.html#warnings.showwarning
        """
        with self._state_lock:
            if self._should_patch_showwarning() and not self._did_patch_showwarning:
                self._original_showwarning = warnings.showwarning
                warnings.showwarning = self._patched_showwarning
                self._did_patch_showwarning = True
            elif not self._should_patch_showwarning() and self._did_patch_showwarning:
                warnings.showwarning = self._original_showwarning
                self._did_patch_showwarning = False

    def set_mlflow_warnings_disablement_state_globally(self, disabled=True):
        """Disables (or re-enables) MLflow warnings globally across all threads.

        Args:
            disabled: If `True`, disables MLflow warnings globally across all threads.
                If `False`, enables MLflow warnings globally across all threads.

        """
        with self._state_lock:
            self._mlflow_warnings_disabled_globally = disabled
            self._modify_patch_state_if_necessary()

    def set_mlflow_warnings_rerouting_state_globally(self, rerouted=True):
        """
        Enables (or disables) rerouting of MLflow warnings to an MLflow event logger with level
        WARNING (e.g. `logger.warning()`) globally across all threads.

        Args:
            rerouted: If `True`, enables MLflow warning rerouting globally across all threads.
                      If `False`, disables MLflow warning rerouting globally across all threads.
        """
        with self._state_lock:
            self._mlflow_warnings_rerouted_to_event_logs = rerouted
            self._modify_patch_state_if_necessary()

    def set_non_mlflow_warnings_disablement_state_for_current_thread(self, disabled=True):
        """Disables (or re-enables) non-MLflow warnings for the current thread.

        Args:
            disabled: If `True`, disables non-MLflow warnings for the current thread. If `False`,
                enables non-MLflow warnings for the current thread. non-MLflow warning
                behavior in other threads is unaffected.

        """
        with self._state_lock:
            if disabled:
                self._disabled_threads.add(get_current_thread_id())
            else:
                self._disabled_threads.discard(get_current_thread_id())
            self._modify_patch_state_if_necessary()

    def set_non_mlflow_warnings_rerouting_state_for_current_thread(self, rerouted=True):
        """Enables (or disables) rerouting of non-MLflow warnings to an MLflow event logger with
        level WARNING (e.g. `logger.warning()`) for the current thread.

        Args:
            rerouted: If `True`, enables non-MLflow warning rerouting for the current thread.
                If `False`, disables non-MLflow warning rerouting for the current thread.
                non-MLflow warning behavior in other threads is unaffected.

        """
        with self._state_lock:
            if rerouted:
                self._rerouted_threads.add(get_current_thread_id())
            else:
                self._rerouted_threads.discard(get_current_thread_id())
            self._modify_patch_state_if_necessary()

    def get_warnings_disablement_state_for_current_thread(self):
        """
        Returns:
            True if non-MLflow warnings are disabled for the current thread. False otherwise.
        """
        return get_current_thread_id() in self._disabled_threads

    def get_warnings_rerouting_state_for_current_thread(self):
        """
        Returns:
            True if non-MLflow warnings are rerouted to an MLflow event logger with level
            WARNING for the current thread. False otherwise.
        """
        return get_current_thread_id() in self._rerouted_threads


_WARNINGS_CONTROLLER = _WarningsController()


class NonMlflowWarningsBehaviorForCurrentThread:
    """
    Context manager that modifies the behavior of non-MLflow warnings upon entry, according to the
    specified parameters.

    Args:
        disable_warnings: If `True`, disable  (mutate & discard) non-MLflow warnings. If `False`,
            do not disable non-MLflow warnings.
        reroute_warnings: If `True`, reroute non-MLflow warnings to an MLflow event logger with
            level WARNING. If `False`, do not reroute non-MLflow warnings.
    """

    def __init__(self, disable_warnings, reroute_warnings):
        """
        Args:
            disable_warnings: If `True`, disable (mutate & discard) non-MLflow warnings.
            reroute_warnings: If `True`, reroute non-MLflow warnings to an MLflow event logger.
        """
        self._disable_warnings = disable_warnings
        self._reroute_warnings = reroute_warnings
        self._prev_disablement_state = None
        self._prev_rerouting_state = None

    def _enter_impl(self):
        self._prev_disablement_state = (
            _WARNINGS_CONTROLLER.get_warnings_disablement_state_for_current_thread()
        )
        self._prev_rerouting_state = (
            _WARNINGS_CONTROLLER.get_warnings_rerouting_state_for_current_thread()
        )

        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_disablement_state_for_current_thread(
            disabled=self._disable_warnings
        )
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_rerouting_state_for_current_thread(
            rerouted=self._reroute_warnings
        )

    def _exit_impl(self, *args, **kwargs):
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_disablement_state_for_current_thread(
            disabled=self._prev_disablement_state
        )
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_rerouting_state_for_current_thread(
            rerouted=self._prev_rerouting_state
        )

    def __enter__(self):
        self._enter_impl()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)
        return None

    async def __aenter__(self):
        self._enter_impl()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)
        return None


class MlflowEventsAndWarningsBehavior:
    """
    Threadsafe context manager that modifies the behavior of MLflow event logging statements
    and MLflow warnings upon entry, according to the specified parameters. Modifications are
    applied globally across all threads and are not reverted until all threads that have made
    a particular modification have exited the context.

    Args:
        disable_event_logs: If `True`, disable (mute & discard) MLflow event logging statements.
            If `False`, do not disable MLflow event logging statements.
        disable_warnings: If `True`, disable  (mutate & discard) MLflow warnings. If `False`,
            do not disable MLflow warnings.
        reroute_warnings: If `True`, reroute MLflow warnings to an MLflow event logger with
            level WARNING. If `False`, do not reroute MLflow warnings.
    """

    _lock = RLock()
    _disable_event_logs_count = 0
    _disable_warnings_count = 0
    _reroute_warnings_count = 0

    def __init__(self, disable_event_logs, disable_warnings, reroute_warnings):
        self._disable_event_logs = disable_event_logs
        self._disable_warnings = disable_warnings
        self._reroute_warnings = reroute_warnings

    def _enter_impl(self):
        try:
            with self._lock:
                if self._disable_event_logs:
                    if self._disable_event_logs_count <= 0:
                        logging_utils.disable_logging()
                    self._disable_event_logs_count += 1

                if self._disable_warnings:
                    if self._disable_warnings_count <= 0:
                        _WARNINGS_CONTROLLER.set_mlflow_warnings_disablement_state_globally(
                            disabled=True
                        )
                    self._disable_warnings_count += 1

                if self._reroute_warnings:
                    if self._reroute_warnings_count <= 0:
                        _WARNINGS_CONTROLLER.set_mlflow_warnings_rerouting_state_globally(
                            rerouted=True
                        )
                    self._reroute_warnings_count += 1
        except Exception:
            pass

    def _exit_impl(self, *args, **kwargs):
        try:
            with self._lock:
                if self._disable_event_logs:
                    self._disable_event_logs_count -= 1
                if self._disable_warnings:
                    self._disable_warnings_count -= 1
                if self._reroute_warnings:
                    self._reroute_warnings_count -= 1

                if self._disable_event_logs_count <= 0:
                    logging_utils.enable_logging()
                if self._disable_warnings_count <= 0:
                    _WARNINGS_CONTROLLER.set_mlflow_warnings_disablement_state_globally(
                        disabled=False
                    )
                if self._reroute_warnings_count <= 0:
                    _WARNINGS_CONTROLLER.set_mlflow_warnings_rerouting_state_globally(
                        rerouted=False
                    )
        except Exception:
            pass

    def __enter__(self):
        self._enter_impl()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)
        return None

    async def __aenter__(self):
        self._enter_impl()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)
        return None
