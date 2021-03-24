import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from threading import RLock, get_ident as get_current_thread_id

import mlflow
import mlflow.utils.logging_utils as logging_utils


class _WarningsController:
    """
    Provides threadsafe utilities to modify warning behavior for MLflow autologging, including:

    - Global disablement of MLflow warnings across all threads
    - Global rerouting of MLflow warnings to an MLflow event logger (i.e. `logger.warn()`)
      across all threads
    - Disablement of all warnings (MLflow and others) for the current thread
    """

    def __init__(self):
        self._mlflow_root_path = Path(os.path.dirname(mlflow.__file__)).resolve()
        self._state_lock = RLock()

        self._did_patch_showwarning = False
        self._original_showwarning = None

        self._silenced_threads = set()
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

        from mlflow.utils.autologging import _logger

        # If the warning's source file is contained within the MLflow package's base
        # directory, it is an MLflow warning and should be emitted via `logger.warning`
        warning_source_path = Path(filename).resolve()
        is_mlflow_warning = self._mlflow_root_path in warning_source_path.parents
        curr_thread = get_current_thread_id()

        if (curr_thread in self._silenced_threads) or (is_mlflow_warning and self._mlflow_warnings_disabled_globally):
            return
        elif (curr_thread in self._rerouted_threads and not is_mlflow_warning) or (is_mlflow_warning and self._mlflow_warnings_rerouted_to_event_logs):
            _logger.warning(
                "MLflow autologging encountered a warning:" ' "%s:%d: %s: %s"',
                filename,
                lineno,
                category.__name__,
                message,
            )
        else:
            self._original_showwarning(message, category, filename, lineno, *args, **kwargs)

    def _should_patch_showwarning(self):
        return (
            (len(self._silenced_threads) > 0)
            or (len(self._rerouted_threads) > 0)
            or self._mlflow_warnings_disabled_globally
            or self._mlflow_warnings_rerouted_to_event_logs
        )

    def _modify_patch_state_if_necessary(self):
        """
        Patches or unpatches `warnings.showwarning` if necessary, as determined by:
            - Whether or not `warnings.showwarning` is already patched
            - Whether or not any custom warning state has been configured on the warnings
              controller (i.e. silencing of all warnings for a particular thread, global
              disablement or rerouting of MLflow warnings)

        Note that reassigning `warnings.showwarning` is the standard / recommended approach for
        modifying warning message display behaviors. For reference, see
        https://docs.python.org/3/library/warnings.html#warnings.showwarning
        """
        with self._state_lock:
            if self._should_patch_showwarning() and not self._did_patch_showwarning:
                self._original_showwarning = warnings.showwarning
                warnings.showwarning = lambda *args, **kwargs: self._patched_showwarning(
                    *args, **kwargs
                )
                self._did_patch_showwarning = True
            elif not self._should_patch_showwarning() and self._did_patch_showwarning:
                warnings.showwarning = self._original_showwarning
                self._did_patch_showwarning = False

    def set_mlflow_warnings_disablement_state_globally(self, disabled=True):
        with self._state_lock:
            self._mlflow_warnings_disabled_globally = disabled
            self._modify_patch_state_if_necessary()

    def set_mlflow_warnings_rerouting_state_globally(self, rerouted=True):
        with self._state_lock:
            self._mlflow_warnings_rerouted_to_event_logs = rerouted
            self._modify_patch_state_if_necessary()

    def set_non_mlflow_warnings_disablement_state_for_current_thread(self, disabled=True):
        with self._state_lock:
            if disabled:
                self._silenced_threads.add(get_current_thread_id())
            else:
                self._silenced_threads.discard(get_current_thread_id())
            self._modify_patch_state_if_necessary()

    def set_non_mlflow_warnings_rerouting_state_for_current_thread(self, rerouted=True):
        with self._state_lock:
            if rerouted:
                self._rerouted_threads.add(get_current_thread_id())
            else:
                self._rerouted_threads.discard(get_current_thread_id())
            self._modify_patch_state_if_necessary()

    def get_warnings_disablement_state_for_current_thread(self):
        return get_current_thread_id() in self._silenced_threads

    def get_warnings_rerouting_state_for_current_thread(self):
        return get_current_thread_id() in self._rerouted_threads


_WARNINGS_CONTROLLER = _WarningsController()


@contextmanager
def set_non_mlflow_warnings_behavior_for_current_thread(disable_warnings, reroute_warnings):
    """
    Context manager that silences (or unsilences) all warnings (MLflow and others) for the current
    thread upon entry. Upon exit, the previous silencing state is restored.
    """
    prev_disablement_state = _WARNINGS_CONTROLLER.get_warnings_disablement_state_for_current_thread()
    prev_rerouting_state = _WARNINGS_CONTROLLER.get_warnings_rerouting_state_for_current_thread()
    try:
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_disablement_state_for_current_thread(disabled=disable_warnings)
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_rerouting_state_for_current_thread(rerouted=reroute_warnings)
        yield
    finally:
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_disablement_state_for_current_thread(
            disabled=prev_disablement_state
        )
        _WARNINGS_CONTROLLER.set_non_mlflow_warnings_rerouting_state_for_current_thread(
            rerouted=prev_rerouting_state
        )


@contextmanager
def set_mlflow_events_and_warnings_behavior_globally(disable_event_logs, disable_warnings, reroute_warnings):
    with _SetMLflowEventsAndWarningsBehaviorGlobally(disable_event_logs, disable_warnings, reroute_warnings):
        yield


class _SetMLflowEventsAndWarningsBehaviorGlobally:
    """
    Threadsafe context manager that silences all MLflow event logging statements and MLflow warnings
    upon entry. Silencing is applied globally across all threads. In single-threaded cases, MLflow
    event logging statements and warnings are re-enabled when the context manager exits. In
    multi-threaded cases, MLflow event logging statements and warnings are re-enabled once all
    instances of this context manager have exited across all threads.
    """

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
                    if  _SetMLflowEventsAndWarningsBehaviorGlobally._disable_event_logs_count <= 0:
                        logging_utils.disable_logging()
                    _SetMLflowEventsAndWarningsBehaviorGlobally._disable_event_logs_count += 1

                if self._disable_warnings:
                    if _SetMLflowEventsAndWarningsBehaviorGlobally._disable_warnings_count <= 0:
                        _WARNINGS_CONTROLLER.set_mlflow_warnings_disablement_state_globally(disabled=True)
                    _SetMLflowEventsAndWarningsBehaviorGlobally._disable_warnings_count += 1

                if self._reroute_warnings:
                    if _SetMLflowEventsAndWarningsBehaviorGlobally._reroute_warnings_count <= 0:
                        _WARNINGS_CONTROLLER.set_mlflow_warnings_rerouting_state_globally(rerouted=True)
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
                    _WARNINGS_CONTROLLER.set_mlflow_warnings_disablement_state_globally(disabled=False)
                if _SetMLflowEventsAndWarningsBehaviorGlobally._reroute_warnings_count <= 0:
                    _WARNINGS_CONTROLLER.set_mlflow_warnings_rerouting_state_globally(rerouted=False)
        except Exception:
            pass
