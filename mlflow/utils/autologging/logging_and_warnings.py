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
        self._mlflow_warnings_disabled_globally = False
        self._mlflow_warnings_rerouted_to_event_logs = False

    def _patched_showwarning(self, message, category, filename, lineno, *args, **kwargs):
        from mlflow.utils.autologging import _logger

        # If the warning's source file is contained within the MLflow package's base
        # directory, it is an MLflow warning and should be emitted via `logger.warning`
        warning_source_path = Path(filename).resolve()
        is_mlflow_warning = self._mlflow_root_path in warning_source_path.parents
        curr_thread = get_current_thread_id()

        if curr_thread in self._silenced_threads:
            return
        elif is_mlflow_warning and self._mlflow_warnings_disabled_globally:
            return
        elif is_mlflow_warning and self._mlflow_warnings_rerouted_to_event_logs:
            _logger.warning(
                "MLflow issued a warning during autologging:" ' "%s:%d: %s: %s"',
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

    def set_global_mlflow_warnings_disablement_state(self, disabled=True):
        """
        Disables (or re-enables) MLflow warnings globally across all threads.

        :param disabled: If `True`, disables MLflow warnings globally across all threads.
                         If `False`, enables MLflow warnings globally across all threads.
        """
        with self._state_lock:
            self._mlflow_warnings_disabled_globally = disabled
            self._modify_patch_state_if_necessary()

    def set_global_mlflow_warnings_rerouting_state(self, rerouted=True):
        """
        Enables (or disables) rerouting of MLflow warnings to an MLflow event logger
        (e.g. `logger.warning()`).

        :param rerouted: If `True`, enables MLflow warning rerouting globally across all threads.
                         If `False`, disables MLflow warning rerouting globally across all threads.
        """
        with self._state_lock:
            self._mlflow_warnings_rerouted_to_event_logs = rerouted
            self._modify_patch_state_if_necessary()

    def set_all_warnings_disablement_state_for_current_thread(self, disabled=True):
        """
        Disables (or re-enables) all warnings (MLflow and others) for the current thread.

        :param disabled: If `True`, disables warnings for the current thread. If `False`, enables
                         MLflow warnings for the current thread. Other threads are unaffected.
        """
        with self._state_lock:
            if disabled:
                self._silenced_threads.add(get_current_thread_id())
            else:
                self._silenced_threads.discard(get_current_thread_id())
            self._modify_patch_state_if_necessary()

    def warnings_disabled_for_current_thread(self):
        """
        :return: `True` if all warnings (MLflow and others) are disabled for the current thread.
                 `False` otherwise.
        """
        return get_current_thread_id() in self._silenced_threads


_WARNINGS_CONTROLLER = _WarningsController()


@contextmanager
def augment_mlflow_warnings():
    """
    MLflow routines called by autologging patch code may issue warnings via the `warnings.warn`
    API. In many cases, the user cannot remediate the cause of these warnings because
    they result from the autologging patch implementation, rather than a user-facing API call.

    Accordingly, this context manager is designed to augment MLflow warnings issued during
    autologging patch code execution, explaining that such warnings were raised as a result of
    MLflow's autologging implementation. MLflow warnings are also redirected from `sys.stderr`
    to an MLflow logger with level WARNING. Warnings issued by code outside of MLflow are
    not modified. When the context manager exits, the original output behavior for MLflow warnings
    is restored.

    Note that the implementation of `augment_mlflow_warnings` is *not* threadsafe.
    """
    try:
        _WARNINGS_CONTROLLER.set_global_mlflow_warnings_rerouting_state(rerouted=True)
        yield
    finally:
        _WARNINGS_CONTROLLER.set_global_mlflow_warnings_rerouting_state(rerouted=False)


@contextmanager
def silence_warnings_and_mlflow_event_logs_if_necessary(autologging_integration):
    """
    Context manager that silences all warnings (MLflow warnings and others) and all MLflow event
    logging statements upon entry if silent mode is enabled for the specified
    `autologging_integration` (i.e. if the `silent` attribute of the integration's configuration is
    `True`). If silent mode is *not* enabled for the specified integration, this context manager is
    a no-op. Upon exit, this context manager re-enables warnings and MLflow event logging
    statements, making a best-effort attempt to provide thread safety (details below).

    When the context manager is active, MLflow event logging statements are silenced *globally
    across all threads*, while warnings (MLflow warnings and others) are silenced for the
    *current thread*. The reasons for this behavior are as follows:

    - During autologging sessions with `silent=True`, all MLflow warnings and MLflow logging
      statements should be silenced in a threadsafe fashion. We should not leak MLflow logging
      statements in multithreaded contexts.

    - During autologging sessions with `silent=True`, non-MLflow warnings and event logging
      statements emitted from original / underlying ML routines should never be suppressed.
      We should not omit any of these warnings and event logging statements in multithreaded
      contexts.

    This context manager attempts to be threadsafe across autologging sessions, with the following
    caveats:

    - If a silent autologging session is run in parallel with a non-silent autologging session,
      MLflow event logging statements and warnings from the non-silent session may be suppressed.

    - If `warnings.showwarning` is unexpectedly re-assigned during the execution of a user's
      original / underlying ML code within an autologging session, warnings may not be
      properly suppressed.

    - If the preamble and postamble of a silent autologging session (i.e. the code surrounding the
      user's original / underlying ML routine) make multithreaded API calls, warnings from
      these API calls may not be suppressed.
    """
    from mlflow.utils.autologging import get_autologging_config

    should_silence = get_autologging_config(autologging_integration, "silent", False)
    if should_silence:
        with _SilenceMLflowEventLogsAndWarningsGlobally(), set_warnings_silence_state_for_current_thread(
            silence=True
        ):
            yield
    else:
        yield


@contextmanager
def set_warnings_silence_state_for_current_thread(silence=True):
    """
    Context manager that silences (or unsilences) all warnings (MLflow and others) for the current
    thread upon entry. Upon exit, the previous silencing state is restored.
    """
    prev_silence_state = _WARNINGS_CONTROLLER.warnings_disabled_for_current_thread()
    try:
        _WARNINGS_CONTROLLER.set_all_warnings_disablement_state_for_current_thread(disabled=silence)
        yield
    finally:
        _WARNINGS_CONTROLLER.set_all_warnings_disablement_state_for_current_thread(
            disabled=prev_silence_state
        )


class _SilenceMLflowEventLogsAndWarningsGlobally:
    """
    Threadsafe context manager that silences all MLflow event logging statements and MLflow warnings
    upon entry. Silencing is applied globally across all threads. In single-threaded cases, MLflow
    event logging statements and warnings are re-enabled when the context manager exits. In
    multi-threaded cases, MLflow event logging statements and warnings are re-enabled once all
    instances of this context manager have exited across all threads.
    """

    _lock = RLock()
    _silenced_count = 0

    def __enter__(self):
        try:
            with _SilenceMLflowEventLogsAndWarningsGlobally._lock:
                if _SilenceMLflowEventLogsAndWarningsGlobally._silenced_count <= 0:
                    logging_utils.disable_logging()
                    _WARNINGS_CONTROLLER.set_global_mlflow_warnings_disablement_state(disabled=True)
                _SilenceMLflowEventLogsAndWarningsGlobally._silenced_count += 1
        except Exception:
            pass

    def __exit__(self, *args, **kwargs):
        try:
            with _SilenceMLflowEventLogsAndWarningsGlobally._lock:
                _SilenceMLflowEventLogsAndWarningsGlobally._silenced_count -= 1
                if _SilenceMLflowEventLogsAndWarningsGlobally._silenced_count <= 0:
                    logging_utils.enable_logging()
                    _WARNINGS_CONTROLLER.set_global_mlflow_warnings_disablement_state(
                        disabled=False
                    )
        except Exception:
            pass
