import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from threading import RLock, get_ident as get_current_thread_id

import mlflux
import mlflux.utils.logging_utils as logging_utils


class _WarningsController:
    """
    Provides threadsafe utilities to modify warning behavior for mlflux autologging, including:

    - Global disablement of mlflux warnings across all threads
    - Global rerouting of mlflux warnings to an mlflux event logger (i.e. `logger.warn()`)
      across all threads
    - Disablement of non-mlflux warnings for the current thread
    - Rerouting of non-mlflux warnings to an mlflux event logger for the current thread
    """

    def __init__(self):
        self._mlflow_root_path = Path(os.path.dirname(mlflux.__file__)).resolve()
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
        options configured on the controller (e.g. rerouting or disablement of mlflux warnings,
        disablement of all warnings for the current thread).

        Note that reassigning `warnings.showwarning` is the standard / recommended approach for
        modifying warning message display behaviors. For reference, see
        https://docs.python.org/3/library/warnings.html#warnings.showwarning
        """
        # NB: We explicitly avoid blocking on the `self._state_lock` lock during `showwarning`
        # to so that threads don't have to execute serially whenever they emit warnings with
        # `warnings.warn()`. We only lock during configuration changes to ensure that
        # `warnings.showwarning` is patched or unpatched at the correct times.

        from mlflux.utils.autologging_utils import _logger

        # If the warning's source file is contained within the mlflux package's base
        # directory, it is an mlflux warning and should be emitted via `logger.warning`
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
                "mlflux autologging encountered a warning:" ' "%s:%d: %s: %s"',
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
        """
        Disables (or re-enables) mlflux warnings globally across all threads.

        :param disabled: If `True`, disables mlflux warnings globally across all threads.
                         If `False`, enables mlflux warnings globally across all threads.
        """
        with self._state_lock:
            self._mlflow_warnings_disabled_globally = disabled
            self._modify_patch_state_if_necessary()

    def set_mlflow_warnings_rerouting_state_globally(self, rerouted=True):
        """
        Enables (or disables) rerouting of mlflux warnings to an mlflux event logger with level
        WARNING (e.g. `logger.warning()`) globally across all threads.

        :param rerouted: If `True`, enables mlflux warning rerouting globally across all threads.
                         If `False`, disables mlflux warning rerouting globally across all threads.
        """
        with self._state_lock:
            self._mlflow_warnings_rerouted_to_event_logs = rerouted
            self._modify_patch_state_if_necessary()

    def set_non_mlflow_warnings_disablement_state_for_current_thread(self, disabled=True):
        """
        Disables (or re-enables) non-mlflux warnings for the current thread.

        :param disabled: If `True`, disables non-mlflux warnings for the current thread. If `False`,
                         enables non-mlflux warnings for the current thread. non-mlflux warning
                         behavior in other threads is unaffected.
        """
        with self._state_lock:
            if disabled:
                self._disabled_threads.add(get_current_thread_id())
            else:
                self._disabled_threads.discard(get_current_thread_id())
            self._modify_patch_state_if_necessary()

    def set_non_mlflow_warnings_rerouting_state_for_current_thread(self, rerouted=True):
        """
        Enables (or disables) rerouting of non-mlflux warnings to an mlflux event logger with level
        WARNING (e.g. `logger.warning()`) for the current thread.

        :param rerouted: If `True`, enables non-mlflux warning rerouting for the current thread.
                         If `False`, disables non-mlflux warning rerouting for the current thread.
                         non-mlflux warning behavior in other threads is unaffected.
        """
        with self._state_lock:
            if rerouted:
                self._rerouted_threads.add(get_current_thread_id())
            else:
                self._rerouted_threads.discard(get_current_thread_id())
            self._modify_patch_state_if_necessary()

    def get_warnings_disablement_state_for_current_thread(self):
        """
        :return: `True` if non-mlflux warnings are disabled for the current thread.
                 `False` otherwise.
        """
        return get_current_thread_id() in self._disabled_threads

    def get_warnings_rerouting_state_for_current_thread(self):
        """
        :return: `True` if non-mlflux warnings are rerouted to an mlflux event logger with level
                 WARNING for the current thread. `False` otherwise.
        """
        return get_current_thread_id() in self._rerouted_threads


_WARNINGS_CONTROLLER = _WarningsController()


@contextmanager
def set_non_mlflow_warnings_behavior_for_current_thread(disable_warnings, reroute_warnings):
    """
    Context manager that modifies the behavior of non-mlflux warnings upon entry, according to the
    specified parameters.

    :param disable_warnings: If `True`, disable  (mutate & discard) non-mlflux warnings. If `False`,
                             do not disable non-mlflux warnings.
    :param reroute_warnings: If `True`, reroute non-mlflux warnings to an mlflux event logger with
                             level WARNING. If `False`, do not reroute non-mlflux warnings.
    """
    prev_disablement_state = (
        _WARNINGS_CONTROLLER.get_warnings_disablement_state_for_current_thread()
    )
    prev_rerouting_state = _WARNINGS_CONTROLLER.get_warnings_rerouting_state_for_current_thread()
    try:
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_disablement_state_for_current_thread(
            disabled=disable_warnings
        )
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_rerouting_state_for_current_thread(
            rerouted=reroute_warnings
        )
        yield
    finally:
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_disablement_state_for_current_thread(
            disabled=prev_disablement_state
        )
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_rerouting_state_for_current_thread(
            rerouted=prev_rerouting_state
        )


@contextmanager
def set_mlflow_events_and_warnings_behavior_globally(
    disable_event_logs, disable_warnings, reroute_warnings
):
    """
    Threadsafe context manager that modifies the behavior of mlflux event logging statements
    and mlflux warnings upon entry, according to the specified parameters. Modifications are
    applied globally across all threads and are not reverted until all threads that have made
    a particular modification have exited the context.

    :param disable_event_logs: If `True`, disable (mute & discard) mlflux event logging statements.
                               If `False`, do not disable mlflux event logging statements.
    :param disable_warnings: If `True`, disable  (mutate & discard) mlflux warnings. If `False`,
                             do not disable mlflux warnings.
    :param reroute_warnings: If `True`, reroute mlflux warnings to an mlflux event logger with
                             level WARNING. If `False`, do not reroute mlflux warnings.
    """

    with _SetMLflowEventsAndWarningsBehaviorGlobally(
        disable_event_logs, disable_warnings, reroute_warnings
    ):
        yield


class _SetMLflowEventsAndWarningsBehaviorGlobally:
    _lock = RLock()
    _disable_event_logs_count = 0
    _disable_warnings_count = 0
    _reroute_warnings_count = 0

    def __init__(self, disable_event_logs, disable_warnings, reroute_warnings):
        self._disable_event_logs = disable_event_logs
        self._disable_warnings = disable_warnings
        self._reroute_warnings = reroute_warnings

    def __enter__(self):
        try:
            with _SetMLflowEventsAndWarningsBehaviorGlobally._lock:
                if self._disable_event_logs:
                    if _SetMLflowEventsAndWarningsBehaviorGlobally._disable_event_logs_count <= 0:
                        logging_utils.disable_logging()
                    _SetMLflowEventsAndWarningsBehaviorGlobally._disable_event_logs_count += 1

                if self._disable_warnings:
                    if _SetMLflowEventsAndWarningsBehaviorGlobally._disable_warnings_count <= 0:
                        _WARNINGS_CONTROLLER.set_mlflow_warnings_disablement_state_globally(
                            disabled=True
                        )
                    _SetMLflowEventsAndWarningsBehaviorGlobally._disable_warnings_count += 1

                if self._reroute_warnings:
                    if _SetMLflowEventsAndWarningsBehaviorGlobally._reroute_warnings_count <= 0:
                        _WARNINGS_CONTROLLER.set_mlflow_warnings_rerouting_state_globally(
                            rerouted=True
                        )
                    _SetMLflowEventsAndWarningsBehaviorGlobally._reroute_warnings_count += 1
        except Exception:
            pass

    def __exit__(self, *args, **kwargs):
        try:
            with _SetMLflowEventsAndWarningsBehaviorGlobally._lock:
                if self._disable_event_logs:
                    _SetMLflowEventsAndWarningsBehaviorGlobally._disable_event_logs_count -= 1
                if self._disable_warnings:
                    _SetMLflowEventsAndWarningsBehaviorGlobally._disable_warnings_count -= 1
                if self._reroute_warnings:
                    _SetMLflowEventsAndWarningsBehaviorGlobally._reroute_warnings_count -= 1

                if _SetMLflowEventsAndWarningsBehaviorGlobally._disable_event_logs_count <= 0:
                    logging_utils.enable_logging()
                if _SetMLflowEventsAndWarningsBehaviorGlobally._disable_warnings_count <= 0:
                    _WARNINGS_CONTROLLER.set_mlflow_warnings_disablement_state_globally(
                        disabled=False
                    )
                if _SetMLflowEventsAndWarningsBehaviorGlobally._reroute_warnings_count <= 0:
                    _WARNINGS_CONTROLLER.set_mlflow_warnings_rerouting_state_globally(
                        rerouted=False
                    )
        except Exception:
            pass
