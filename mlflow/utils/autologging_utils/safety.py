import abc
import functools
import inspect
import itertools
import typing
import uuid
from abc import abstractmethod
from contextlib import contextmanager

import mlflow
import mlflow.utils.autologging_utils
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import _MLFLOW_AUTOLOGGING_TESTING
from mlflow.tracking.client import MlflowClient
from mlflow.utils import gorilla, is_iterator
from mlflow.utils.autologging_utils import _logger
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
    set_mlflow_events_and_warnings_behavior_globally,
    set_non_mlflow_warnings_behavior_for_current_thread,
)
from mlflow.utils.mlflow_tags import MLFLOW_AUTOLOGGING

_AUTOLOGGING_PATCHES = {}


# Function attribute used for testing purposes to verify that a given function
# has been wrapped with the `exception_safe_function_for_class` and
# `picklable_exception_safe_function` decorators
_ATTRIBUTE_EXCEPTION_SAFE = "exception_safe"


def exception_safe_function_for_class(function):
    """
    Wraps the specified function with broad exception handling to guard
    against unexpected errors during autologging.
    Note this function creates an unpicklable function as `safe_function` is locally defined,
    but a class instance containing methods decorated by this function should be pickalable,
    because pickle only saves instance attributes, not methods.
    See https://docs.python.org/3/library/pickle.html#pickling-class-instances for more details.
    """
    if is_testing():
        setattr(function, _ATTRIBUTE_EXCEPTION_SAFE, True)

    def safe_function(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            if is_testing():
                raise
            else:
                _logger.warning("Encountered unexpected error during autologging: %s", e)

    return update_wrapper_extended(safe_function, function)


def _safe_function(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as e:
        if is_testing():
            raise
        else:
            _logger.warning("Encountered unexpected error during autologging: %s", e)


def picklable_exception_safe_function(function):
    """
    Wraps the specified function with broad exception handling to guard
    against unexpected errors during autologging while preserving picklability.
    """
    if is_testing():
        setattr(function, _ATTRIBUTE_EXCEPTION_SAFE, True)

    return update_wrapper_extended(functools.partial(_safe_function, function), function)


def _exception_safe_class_factory(base_class):
    """
    Creates an exception safe metaclass that inherits from `base_class`.
    """

    class _ExceptionSafeClass(base_class):
        """
        Metaclass that wraps all functions defined on the specified class with broad error handling
        logic to guard against unexpected errors during autlogging.

        Rationale: Patched autologging functions commonly pass additional class instances as
        arguments to their underlying original training routines; for example, Keras autologging
        constructs a subclass of `keras.callbacks.Callback` and forwards it to `Model.fit()`.
        To prevent errors encountered during method execution within such classes from disrupting
        model training, this metaclass wraps all class functions in a broad try / catch statement.

        Note: `ExceptionSafeClass` does not handle exceptions in class methods or static methods,
        as these are not always Python callables and are difficult to wrap
        """

        def __new__(cls, name, bases, dct):
            for m in dct:
                # class methods or static methods are not callable.
                if callable(dct[m]):
                    dct[m] = exception_safe_function_for_class(dct[m])
            return base_class.__new__(cls, name, bases, dct)

    return _ExceptionSafeClass


ExceptionSafeClass = _exception_safe_class_factory(type)

# `ExceptionSafeClass` causes an error when used with an abstract class.
#
# ```
# class AbstractClass(abc.ABC):
#    ...
#
# class DerivedClass(AbstractClass, metaclass=ExceptionSafeClass):
#    ...
# ```
#
# This raises:
#
# ```
# TypeError: metaclass conflict: the metaclass of a derived class must be
#            a (non-strict) subclass of the metaclasses of all its bases.
# ```
#
# To avoid this error, create `ExceptionSafeAbstractClass` that is based on `abc.ABCMeta`.
ExceptionSafeAbstractClass = _exception_safe_class_factory(abc.ABCMeta)


class PatchFunction:
    """
    Base class representing a function patch implementation with a callback for error handling.
    `PatchFunction` should be subclassed and used in conjunction with `safe_patch` to
    safely modify the implementation of a function. Subclasses of `PatchFunction` should
    use `_patch_implementation` to define modified ("patched") function implementations and
    `_on_exception` to define cleanup logic when `_patch_implementation` terminates due
    to an unhandled exception.
    """

    @abstractmethod
    def _patch_implementation(self, original, *args, **kwargs):
        """
        Invokes the patch function code.

        Args:
            original: The original, underlying function over which the `PatchFunction`
                is being applied.
            *args: The positional arguments passed to the original function.
            **kwargs: The keyword arguments passed to the original function.
        """

    @abstractmethod
    def _on_exception(self, exception):
        """Called when an unhandled standard Python exception (i.e. an exception inheriting from
        `Exception`) or a `KeyboardInterrupt` prematurely terminates the execution of
        `_patch_implementation`.

        Args:
            exception: The unhandled exception thrown by `_patch_implementation`.
        """

    @classmethod
    def call(cls, original, *args, **kwargs):
        return cls().__call__(original, *args, **kwargs)

    def __call__(self, original, *args, **kwargs):
        try:
            return self._patch_implementation(original, *args, **kwargs)
        except (Exception, KeyboardInterrupt) as e:
            try:
                self._on_exception(e)
            finally:
                # Regardless of what happens during the `_on_exception` callback, reraise
                # the original implementation exception once the callback completes
                raise e


def with_managed_run(autologging_integration, patch_function, tags=None):
    """Given a `patch_function`, returns an `augmented_patch_function` that wraps the execution of
    `patch_function` with an active MLflow run. The following properties apply:

        - An MLflow run is only created if there is no active run present when the
          patch function is executed

        - If an active run is created by the `augmented_patch_function`, it is terminated
          with the `FINISHED` state at the end of function execution

        - If an active run is created by the `augmented_patch_function`, it is terminated
          with the `FAILED` if an unhandled exception is thrown during function execution

    Note that, if nested runs or non-fluent runs are created by `patch_function`, `patch_function`
    is responsible for terminating them by the time it terminates
    (or in the event of an exception).

    Args:
        autologging_integration: The autologging integration associated
            with the `patch_function`.
        patch_function: A `PatchFunction` class definition or a function object
            compatible with `safe_patch`.
        tags: A dictionary of string tags to set on each managed run created during the
            execution of `patch_function`.
    """

    def create_managed_run():
        managed_run = mlflow.start_run(tags=tags)
        _logger.info(
            "Created MLflow autologging run with ID '%s', which will track hyperparameters,"
            " performance metrics, model artifacts, and lineage information for the"
            " current %s workflow",
            managed_run.info.run_id,
            autologging_integration,
        )
        return managed_run

    if inspect.isclass(patch_function):

        class PatchWithManagedRun(patch_function):
            def __init__(self):
                super().__init__()
                self.managed_run = None

            def _patch_implementation(self, original, *args, **kwargs):
                if not mlflow.active_run():
                    self.managed_run = create_managed_run()

                result = super()._patch_implementation(original, *args, **kwargs)

                if self.managed_run:
                    mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))

                return result

            def _on_exception(self, e):
                if self.managed_run:
                    mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))
                super()._on_exception(e)

        return PatchWithManagedRun

    else:

        def patch_with_managed_run(original, *args, **kwargs):
            managed_run = None
            if not mlflow.active_run():
                managed_run = create_managed_run()

            try:
                result = patch_function(original, *args, **kwargs)
            except (Exception, KeyboardInterrupt):
                # In addition to standard Python exceptions, handle keyboard interrupts to ensure
                # that runs are terminated if a user prematurely interrupts training execution
                # (e.g. via sigint / ctrl-c)
                if managed_run:
                    mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))
                raise
            else:
                if managed_run:
                    mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))
                return result

        return patch_with_managed_run


def is_testing():
    """
    Indicates whether or not autologging functionality is running in test mode (as determined
    by the `MLFLOW_AUTOLOGGING_TESTING` environment variable). Test mode performs additional
    validation during autologging, including:

        - Checks for the exception safety of arguments passed to model training functions
          (i.e. all additional arguments should be "exception safe" functions or classes)
        - Disables exception handling for patched function logic, ensuring that patch code
          executes without errors during testing
    """
    return _MLFLOW_AUTOLOGGING_TESTING.get()


def _resolve_extra_tags(autologging_integration, extra_tags):
    tags = {MLFLOW_AUTOLOGGING: autologging_integration}
    if extra_tags:
        if isinstance(extra_tags, dict):
            if MLFLOW_AUTOLOGGING in extra_tags:
                extra_tags.pop(MLFLOW_AUTOLOGGING)
                _logger.warning(
                    f"Tag `{MLFLOW_AUTOLOGGING}` is ignored as it is a reserved tag by MLflow "
                    f"autologging."
                )
            tags.update(extra_tags)
        else:
            raise mlflow.exceptions.MlflowException.invalid_parameter_value(
                f"Invalid `extra_tags` type: expecting dictionary, "
                f"received `{type(extra_tags).__name__}`"
            )
    return tags


def safe_patch(
    autologging_integration,
    destination,
    function_name,
    patch_function,
    manage_run=False,
    extra_tags=None,
):
    """Patches the specified `function_name` on the specified `destination` class for autologging
    purposes, preceding its implementation with an error-safe copy of the specified patch
    `patch_function` with the following error handling behavior:
        - Exceptions thrown from the underlying / original function
          (`<destination>.<function_name>`) are propagated to the caller.
        - Exceptions thrown from other parts of the patched implementation (`patch_function`)
          are caught and logged as warnings.

    Args:
        autologging_integration: The name of the autologging integration associated with the
            patch.
        destination: The Python class on which the patch is being defined.
        function_name: The name of the function to patch on the specified `destination` class.
        patch_function: The patched function code to apply. This is either a `PatchFunction`
            class definition or a function object. If it is a function object, the
            first argument should be reserved for an `original` method argument
            representing the underlying / original function. Subsequent arguments
            should be identical to those of the original function being patched.
        manage_run: If `True`, applies the `with_managed_run` wrapper to the specified
            `patch_function`, which automatically creates & terminates an MLflow
            active run during patch code execution if necessary. If `False`,
            does not apply the `with_managed_run` wrapper to the specified
            `patch_function`.
        extra_tags: A dictionary of extra tags to set on each managed run created by autologging.
    """
    from mlflow.utils.autologging_utils import autologging_is_disabled, get_autologging_config

    if manage_run:
        tags = _resolve_extra_tags(autologging_integration, extra_tags)
        patch_function = with_managed_run(
            autologging_integration,
            patch_function,
            tags=tags,
        )

    patch_is_class = inspect.isclass(patch_function)
    if patch_is_class:
        assert issubclass(patch_function, PatchFunction)
    else:
        assert callable(patch_function)

    original_fn = gorilla.get_original_attribute(
        destination, function_name, bypass_descriptor_protocol=False
    )
    # Retrieve raw attribute while bypassing the descriptor protocol
    raw_original_obj = gorilla.get_original_attribute(
        destination, function_name, bypass_descriptor_protocol=True
    )
    if original_fn != raw_original_obj:
        raise RuntimeError(f"Unsupport patch on {destination}.{function_name}")
    elif isinstance(original_fn, property):
        is_property_method = True

        # For property decorated methods (a kind of method delegation), e.g.
        # class A:
        #   @property
        #   def f1(self):
        #     ...
        #     return delegated_f1
        #
        # suppose `a1` is an instance of class `A`,
        # `A.f1.fget` will get the original `def f1(self)` method,
        # and `A.f1.fget(a1)` will be equivalent to `a1.f1()` and
        # its return value will be the `delegated_f1` function.
        # So using the `property.fget` we can construct the (delegated) "original_fn"
        def original(self, *args, **kwargs):
            # the `original_fn.fget` will get the original method decorated by `property`
            # the `original_fn.fget(self)` will get the delegated function returned by the
            # property decorated method.
            bound_delegate_method = original_fn.fget(self)
            return bound_delegate_method(*args, **kwargs)

    else:
        original = original_fn
        is_property_method = False

    def safe_patch_function(*args, **kwargs):
        """
        A safe wrapper around the specified `patch_function` implementation designed to
        handle exceptions thrown during the execution of `patch_function`. This wrapper
        distinguishes exceptions thrown from the underlying / original function
        (`<destination>.<function_name>`) from exceptions thrown from other parts of
        `patch_function`. This distinction is made by passing an augmented version of the
        underlying / original function to `patch_function` that uses nonlocal state to track
        whether or not it has been executed and whether or not it threw an exception.
        Exceptions thrown from the underlying / original function are propagated to the caller,
        while exceptions thrown from other parts of `patch_function` are caught and logged as
        warnings.
        """
        # Reroute warnings encountered during the patch function implementation to an MLflow event
        # logger, and enforce silent mode if applicable (i.e. if the corresponding autologging
        # integration was called with `silent=True`), hiding MLflow event logging statements and
        # hiding all warnings in the autologging preamble and postamble (i.e. the code surrounding
        # the user's original / underlying ML function). Non-MLflow warnings are enabled during the
        # execution of the original / underlying ML function
        #
        # Note that we've opted *not* to apply this context manager as a decorator on
        # `safe_patch_function` because the context-manager-as-decorator pattern uses
        # `contextlib.ContextDecorator`, which creates generator expressions that cannot be pickled
        # during model serialization by ML frameworks such as scikit-learn
        is_silent_mode = get_autologging_config(autologging_integration, "silent", False)
        with set_mlflow_events_and_warnings_behavior_globally(
            # MLflow warnings emitted during autologging training sessions are likely not
            # actionable and result from the autologging implementation invoking another MLflow
            # API. Accordingly, we reroute these warnings to the MLflow event logger with level
            # WARNING For reference, see recommended warning and event logging behaviors from
            # https://docs.python.org/3/howto/logging.html#when-to-use-logging
            reroute_warnings=True,
            disable_event_logs=is_silent_mode,
            disable_warnings=is_silent_mode,
        ), set_non_mlflow_warnings_behavior_for_current_thread(
            # non-MLflow Warnings emitted during the autologging preamble (before the original /
            # underlying ML function is called) and postamble (after the original / underlying ML
            # function is called) are likely not actionable and result from the autologging
            # implementation invoking an API from a dependent library. Accordingly, we reroute
            # these warnings to the MLflow event logger with level WARNING. For reference, see
            # recommended warning and event logging behaviors from
            # https://docs.python.org/3/howto/logging.html#when-to-use-logging
            reroute_warnings=True,
            disable_warnings=is_silent_mode,
        ):
            if is_testing():
                preexisting_run_for_testing = mlflow.active_run()

            # Whether or not to exclude autologged content from user-created fluent runs
            # (i.e. runs created manually via `mlflow.start_run()`)
            exclusive = get_autologging_config(autologging_integration, "exclusive", False)
            user_created_fluent_run_is_active = (
                mlflow.active_run() and not _AutologgingSessionManager.active_session()
            )
            active_session_failed = (
                _AutologgingSessionManager.active_session() is not None
                and _AutologgingSessionManager.active_session().state == "failed"
            )

            if (
                active_session_failed
                or autologging_is_disabled(autologging_integration)
                or (user_created_fluent_run_is_active and exclusive)
                or (
                    mlflow.utils.autologging_utils._AUTOLOGGING_GLOBALLY_DISABLED
                    and autologging_integration
                    not in mlflow.utils.autologging_utils._AUTOLOGGING_GLOBALLY_DISABLED_EXEMPTIONS
                )
            ):
                # If the autologging integration associated with this patch is disabled,
                # or if the current autologging integration is in exclusive mode and a user-created
                # fluent run is active, call the original function and return. Restore the original
                # warning behavior during original function execution, since autologging is being
                # skipped
                with set_non_mlflow_warnings_behavior_for_current_thread(
                    disable_warnings=False,
                    reroute_warnings=False,
                ):
                    return original(*args, **kwargs)

            # Whether or not the original / underlying function has been called during the
            # execution of patched code
            original_has_been_called = False
            # The value returned by the call to the original / underlying function during
            # the execution of patched code
            original_result = None
            # Whether or not an exception was raised from within the original / underlying function
            # during the execution of patched code
            failed_during_original = False
            # The active MLflow run (if any) associated with patch code execution
            patch_function_run_for_testing = None
            # The exception raised during executing patching function
            patch_function_exception = None

            def try_log_autologging_event(log_fn, *args):
                try:
                    log_fn(*args)
                except Exception as e:
                    _logger.debug(
                        "Failed to log autologging event via '%s'. Exception: %s",
                        log_fn,
                        e,
                    )

            def call_original_fn_with_event_logging(original_fn, og_args, og_kwargs):
                try:
                    try_log_autologging_event(
                        AutologgingEventLogger.get_logger().log_original_function_start,
                        session,
                        destination,
                        function_name,
                        og_args,
                        og_kwargs,
                    )
                    original_fn_result = original_fn(*og_args, **og_kwargs)

                    try_log_autologging_event(
                        AutologgingEventLogger.get_logger().log_original_function_success,
                        session,
                        destination,
                        function_name,
                        og_args,
                        og_kwargs,
                    )
                    return original_fn_result
                except Exception as original_fn_e:
                    try_log_autologging_event(
                        AutologgingEventLogger.get_logger().log_original_function_error,
                        session,
                        destination,
                        function_name,
                        og_args,
                        og_kwargs,
                        original_fn_e,
                    )

                    nonlocal failed_during_original
                    failed_during_original = True
                    raise

            with _AutologgingSessionManager.start_session(autologging_integration) as session:
                try:

                    def call_original(*og_args, **og_kwargs):
                        def _original_fn(*_og_args, **_og_kwargs):
                            if is_testing():
                                _validate_args(
                                    autologging_integration,
                                    function_name,
                                    args,
                                    kwargs,
                                    og_args,
                                    og_kwargs,
                                )
                                # By the time `original` is called by the patch implementation, we
                                # assume that either: 1. the patch implementation has already
                                # created an MLflow run or 2. the patch code will not create an
                                # MLflow run during the current execution. Here, we capture a
                                # reference to the active run, which we will use later on to
                                # determine whether or not the patch implementation created
                                # a run and perform validation if necessary
                                nonlocal patch_function_run_for_testing
                                patch_function_run_for_testing = mlflow.active_run()

                            nonlocal original_has_been_called
                            original_has_been_called = True

                            nonlocal original_result
                            # Show all non-MLflow warnings as normal (i.e. not as event logs)
                            # during original function execution, even if silent mode is enabled
                            # (`silent=True`), since these warnings originate from the ML framework
                            # or one of its dependencies and are likely relevant to the caller
                            with set_non_mlflow_warnings_behavior_for_current_thread(
                                disable_warnings=False,
                                reroute_warnings=False,
                            ):
                                original_result = original(*_og_args, **_og_kwargs)
                                return original_result

                        return call_original_fn_with_event_logging(_original_fn, og_args, og_kwargs)

                    # Apply the name, docstring, and signature of `original` to `call_original`.
                    # This is important because several autologging patch implementations inspect
                    # the signature of the `original` argument during execution
                    call_original = update_wrapper_extended(call_original, original)

                    try_log_autologging_event(
                        AutologgingEventLogger.get_logger().log_patch_function_start,
                        session,
                        destination,
                        function_name,
                        args,
                        kwargs,
                    )

                    if patch_is_class:
                        patch_function.call(call_original, *args, **kwargs)
                    else:
                        patch_function(call_original, *args, **kwargs)

                    session.state = "succeeded"

                    try_log_autologging_event(
                        AutologgingEventLogger.get_logger().log_patch_function_success,
                        session,
                        destination,
                        function_name,
                        args,
                        kwargs,
                    )
                except Exception as e:
                    session.state = "failed"
                    patch_function_exception = e
                    # Exceptions thrown during execution of the original function should be
                    # propagated to the caller. Additionally, exceptions encountered during test
                    # mode should be reraised to detect bugs in autologging implementations
                    if failed_during_original or is_testing():
                        raise

                if is_testing() and not preexisting_run_for_testing:
                    # If an MLflow run was created during the execution of patch code, verify that
                    # it is no longer active and that it contains expected autologging tags
                    assert (
                        not mlflow.active_run()
                    ), f"Autologging integration {autologging_integration} leaked an active run"
                    if patch_function_run_for_testing:
                        _validate_autologging_run(
                            autologging_integration, patch_function_run_for_testing.info.run_id
                        )
                try:
                    if original_has_been_called:
                        return original_result
                    else:
                        return call_original_fn_with_event_logging(original, args, kwargs)
                finally:
                    # If original function succeeds, but `patch_function_exception` exists,
                    # it represent patching code unexpected failure, so we call
                    # `log_patch_function_error` in this case.
                    # If original function failed, we don't call `log_patch_function_error`
                    # even if `patch_function_exception` exists, because original function failure
                    # means there's some error in user code (e.g. user provide wrong arguments)
                    if patch_function_exception is not None and not failed_during_original:
                        try_log_autologging_event(
                            AutologgingEventLogger.get_logger().log_patch_function_error,
                            session,
                            destination,
                            function_name,
                            args,
                            kwargs,
                            patch_function_exception,
                        )

                        _logger.warning(
                            "Encountered unexpected error during %s autologging: %s",
                            autologging_integration,
                            patch_function_exception,
                        )

    if is_property_method:
        # Create a patched function (also property decorated)
        # like:
        #
        # class A:
        # @property
        # def get_bound_safe_patch_fn(self):
        #   original_fn.fget(self) # do availability check
        #   return bound_safe_patch_fn
        #
        # Suppose `a1` is instance of class A,
        # then `a1.get_bound_safe_patch_fn(*args, **kwargs)` will be equivalent to
        # `bound_safe_patch_fn(*args, **kwargs)`
        def get_bound_safe_patch_fn(self):
            # This `original_fn.fget` call is for availability check, if it raise error
            # then `hasattr(obj, {func_name})` will return False
            # so it mimic the original property behavior.
            original_fn.fget(self)

            def bound_safe_patch_fn(*args, **kwargs):
                return safe_patch_function(self, *args, **kwargs)

            # Make bound method `instance.target_method` keep the same doc and signature.
            # Here return the bound safe patch function because user call property decorated
            # method will like `instance.property_decorated_method(...)`, and internally it will
            # call the `bound_safe_patch_fn`, the argument list don't include the `self` argument,
            # so return bound function here.
            return update_wrapper_extended(bound_safe_patch_fn, original_fn.fget)

        # Make unbound method `class.target_method` keep the same doc and signature
        get_bound_safe_patch_fn = update_wrapper_extended(get_bound_safe_patch_fn, original_fn.fget)
        safe_patch_obj = property(get_bound_safe_patch_fn)
    else:
        safe_patch_obj = update_wrapper_extended(safe_patch_function, original)

    new_patch = _wrap_patch(destination, function_name, safe_patch_obj)
    _store_patch(autologging_integration, new_patch)


def revert_patches(autologging_integration):
    """Reverts all patches on the specified destination class for autologging disablement purposes.

    Args:
        autologging_integration: The name of the autologging integration associated with the
            patch. Note: If called via fluent api (`autologging_integration="mlflow"`), then revert
            all patches for all active autologging integrations.

    """
    for patch in _AUTOLOGGING_PATCHES.get(autologging_integration, []):
        gorilla.revert(patch)

    _AUTOLOGGING_PATCHES.pop(autologging_integration, None)


# Represents an active autologging session using two fields:
# - integration: the name of the autologging integration corresponding to the session
# - id: a unique session identifier (e.g., a UUID)
# - state: the state of AutologgingSession, will be one of running/succeeded/failed
class AutologgingSession:
    def __init__(self, integration, id_):
        self.integration = integration
        self.id = id_
        self.state = "running"


class _AutologgingSessionManager:
    _session = None

    @classmethod
    @contextmanager
    def start_session(cls, integration):
        try:
            prev_session = cls._session
            if prev_session is None:
                session_id = uuid.uuid4().hex
                cls._session = AutologgingSession(integration, session_id)
            yield cls._session
        finally:
            # Only end the session upon termination of the context if we created
            # the session; otherwise, leave the session open for later termination
            # by its creator
            if prev_session is None:
                cls._end_session()

    @classmethod
    def active_session(cls):
        return cls._session

    @classmethod
    def _end_session(cls):
        cls._session = None


def update_wrapper_extended(wrapper, wrapped):
    """Update a `wrapper` function to look like the `wrapped` function. This is an extension of
    `functools.update_wrapper` that applies the docstring *and* signature of `wrapped` to
    `wrapper`, producing a new function.

    Returns:
        A new function with the same implementation as `wrapper` and the same docstring
        & signature as `wrapped`.
    """
    updated_wrapper = functools.update_wrapper(wrapper, wrapped)
    # Assign the signature of the `wrapped` function to the updated wrapper function.
    # Certain frameworks may disallow signature inspection, causing `inspect.signature()` to throw.
    # One such example is the `tensorflow.estimator.Estimator.export_savedmodel()` function
    try:
        updated_wrapper.__signature__ = inspect.signature(wrapped)
    except Exception:
        _logger.debug("Failed to restore original signature for wrapper around %s", wrapped)
    return updated_wrapper


def _wrap_patch(destination, name, patch_obj, settings=None):
    """Apply a patch.

    Args:
        destination: Patch destination.
        name: Name of the attribute at the destination.
        patch_obj: Patch object, it should be a function or a property decorated function
            to be assigned to the patch point {destination}.{name}.
        settings: Settings for gorilla.Patch.

    """
    if settings is None:
        settings = gorilla.Settings(allow_hit=True, store_hit=True)

    patch = gorilla.Patch(destination, name, patch_obj, settings=settings)
    gorilla.apply(patch)
    return patch


def _store_patch(autologging_integration, patch):
    """
    Stores a patch for a specified autologging_integration class. Later to be used for being able
    to revert the patch when disabling autologging.

    Args:
        autologging_integration: The name of the autologging integration associated with the
            patch.
        patch: The patch to be stored.
    """
    if autologging_integration in _AUTOLOGGING_PATCHES:
        _AUTOLOGGING_PATCHES[autologging_integration].add(patch)
    else:
        _AUTOLOGGING_PATCHES[autologging_integration] = {patch}


def _validate_autologging_run(autologging_integration, run_id):
    """
    For testing purposes, verifies that an MLflow run produced by an `autologging_integration`
    satisfies the following properties:

        - The run has an autologging tag whose value is the name of the autologging integration
        - The run has a terminal status (e.g., KILLED, FAILED, FINISHED)
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    autologging_tag_value = run.data.tags.get(MLFLOW_AUTOLOGGING)
    assert autologging_tag_value == autologging_integration, (
        f"Autologging run with id {run_id} failed to set autologging tag with expected value. "
        f"Expected: '{autologging_integration}', Actual: '{autologging_tag_value}'"
    )
    assert RunStatus.is_terminated(
        RunStatus.from_string(run.info.status)
    ), f"Autologging run with id {run_id} has a non-terminal status '{run.info.status}'"


class ValidationExemptArgument(typing.NamedTuple):
    """
    A NamedTuple representing the properties of an argument that is exempt from validation

    autologging_integration: The name of the autologging integration.
    function_name: The name of the function that is being validated.
    type_function: A Callable that accepts an object and returns True if the given object matches
                   the argument type. Returns False otherwise.
    positional_argument_index: The index of the argument in the function signature.
    keyword_argument_name: The name of the argument in the function signature.
    """

    autologging_integration: str
    function_name: str
    type_function: typing.Callable
    positional_argument_index: int = None
    keyword_argument_name: str = None

    def matches(
        self,
        autologging_integration,
        function_name,
        value,
        argument_index=None,
        argument_name=None,
    ):
        """
        This method checks if the properties provided through the function arguments matches the
        properties defined in the NamedTuple.

        Args:
            autologging_integration: The name of an autologging integration.
            function_name: The name of the function that is being matched.
            value: The value of the argument.
            argument_index: The index of the argument, if it is passed as a positional
                argument. Otherwise it is None.
            argument_name: The name of the argument, if it is passed as a keyword
                argument. Otherwise it is None.

        Returns:
            Returns True if the given function properties matches the exempt argument's
            properties. Returns False otherwise.
        """
        return (
            self.autologging_integration == autologging_integration
            and self.function_name == function_name
            and (
                self.positional_argument_index == argument_index
                or self.keyword_argument_name == argument_name
            )
            and self.type_function(value)
        )


# WARNING: Exemptions should NOT be introduced unless absolutely necessary. If deemed necessary,
#          clear reasons must be provided as comment in addition to thorough integration tests.
_VALIDATION_EXEMPT_ARGUMENTS = [
    # When extracting implicitly defined `batch_size` in the case that `x` is a generator or a
    # generator class, we need to consume and restore the first element back to the generator to
    # calculate the `batch_size`. This means that:
    # 1. The type of `x` will become 'generator' regardless if user provided `x` as a generator or a
    #    custom generator class.
    # 2. The instance of `x` will be different, since we reconstructed the generator after consuming
    #    the first element.
    ValidationExemptArgument("tensorflow", "fit", is_iterator, 1, "x"),
    ValidationExemptArgument("keras", "fit", is_iterator, 1, "x"),
]


def _is_arg_exempt_from_validation(
    autologging_integration,
    function_name,
    argument,
    argument_index=None,
    argument_name=None,
):
    """This function is responsible for determining whether or not an argument is exempt from
    autolog safety validations. This includes both type checking and immutable checking.

    Args:
        autologging_integration: The name of the autologging integration.
        function_name: The name of the function that is being validated.
        argument: The actual argument.
        argument_index: The index of the argument, if it is passed as a positional
            argument. Otherwise it is None.
        argument_name: The name of the argument, if it is passed as a keyword argument.
            Otherwise it is None.

    Returns:
        True or False
    """
    return any(
        exemption.matches(
            autologging_integration,
            function_name,
            argument,
            argument_index,
            argument_name,
        )
        for exemption in _VALIDATION_EXEMPT_ARGUMENTS
    )


def _validate_args(
    autologging_integration,
    function_name,
    user_call_args,
    user_call_kwargs,
    autologging_call_args,
    autologging_call_kwargs,
):
    """
    Used for testing purposes to verify that, when a patched ML function calls its underlying
    / original ML function, the following properties are satisfied:

        - All arguments supplied to the patched ML function are forwarded to the
          original ML function
        - Any additional arguments supplied to the original function are exception safe (i.e.
          they are either functions decorated with the `@exception_safe_function_for_class` or
          `@pickalable_exception_safe_function` decorators, or classes / instances of classes with
          type `ExceptionSafeClass`
    """

    def _validate_new_input(inp):
        """
        Validates a new input (arg or kwarg) introduced to the underlying / original ML function
        call during the execution of a patched ML function. The new input is valid if:

            - The new input is a function that has been decorated with
              `exception_safe_function_for_class` or `pickalable_exception_safe_function`
            - OR the new input is a class with the `ExceptionSafeClass` metaclass
            - OR the new input is a list and each of its elements is valid according to the
              these criteria
        """
        if type(inp) == list:
            for item in inp:
                _validate_new_input(item)
        elif isinstance(inp, dict) and "callbacks" in inp:
            _validate_new_input(inp["callbacks"])
        elif callable(inp):
            assert getattr(inp, _ATTRIBUTE_EXCEPTION_SAFE, False), (
                f"New function argument '{inp}' passed to original function is not exception-safe."
                " Please decorate the function with `exception_safe_function` or "
                "`pickalable_exception_safe_function`"
            )
        else:
            assert hasattr(inp, "__class__") and type(inp.__class__) in [
                ExceptionSafeClass,
                ExceptionSafeAbstractClass,
            ], (
                f"Invalid new input '{inp}'. New args / kwargs introduced to `original` function "
                "calls by patched code must either be functions decorated with "
                "`exception_safe_function_for_class`, instances of classes with the "
                "`ExceptionSafeClass` or `ExceptionSafeAbstractClass` metaclass safe or lists of "
                "such exception safe functions / classes."
            )

    def _assert_autologging_input_positional_args_are_superset(
        autologging_call_input, user_call_input
    ):
        length_diff = len(autologging_call_input) - len(user_call_input)
        assert (
            length_diff >= 0
        ), f"{length_diff} expected inputs are missing from the call to the original function."

    def _assert_autologging_input_kwargs_are_superset(autologging_call_input, user_call_input):
        assert set(user_call_input.keys()).issubset(set(autologging_call_input.keys())), (
            "Keyword or dictionary arguments to original function omit"
            " one or more expected keys: '{}'".format(
                set(user_call_input.keys()) - set(autologging_call_input.keys())
            )
        )

    def _validate(autologging_call_input, user_call_input=None):
        """
        Validates that the specified `autologging_call_input` and `user_call_input`
        are compatible. If `user_call_input` is `None`, then `autologging_call_input`
        is regarded as a new input added by autologging and is validated using
        `_validate_new_input`. Otherwise, the following properties must hold:

            - `autologging_call_input` and `user_call_input` must have the same type
              (referred to as "input type")
            - if the input type is a tuple, list or dictionary, then `autologging_call_input` must
              be equivalent to `user_call_input` or be a superset of `user_call_input`
            - for all other input types, `autologging_call_input` and `user_call_input`
              must be equivalent by reference equality or by object equality

        Args:
            autologging_call_input: call input from autologging.
            user_call_input: call input from user.
        """

        if user_call_input is None and autologging_call_input is not None:
            _validate_new_input(autologging_call_input)
            return

        assert type(autologging_call_input) == type(
            user_call_input
        ), "Type of input to original function '{}' does not match expected type '{}'".format(
            type(autologging_call_input), type(user_call_input)
        )

        if type(autologging_call_input) in [list, tuple]:
            _assert_autologging_input_positional_args_are_superset(
                autologging_call_input, user_call_input
            )
            # If the autologging call input is longer than the user call input, we `zip_longest`
            # will pad the user call input with `None` values to ensure that the subsequent calls
            # to `_validate` identify new inputs added by the autologging call
            for a, u in itertools.zip_longest(autologging_call_input, user_call_input):
                _validate(a, u)
        elif type(autologging_call_input) == dict:
            _assert_autologging_input_kwargs_are_superset(autologging_call_input, user_call_input)
            for key in autologging_call_input.keys():
                _validate(autologging_call_input[key], user_call_input.get(key, None))
        else:
            assert (
                autologging_call_input is user_call_input
                or autologging_call_input == user_call_input
            ), (
                "Input to original function does not match expected input."
                f" Original: '{autologging_call_input}'. Expected: '{user_call_input}'"
            )

    # Similar validation logic found in _validate, unraveling the list of arguments to exclude
    # checks for any validation exempt positional arguments.
    _assert_autologging_input_positional_args_are_superset(autologging_call_args, user_call_args)
    for index, autologging_call_arg, user_call_arg in itertools.zip_longest(
        range(len(user_call_args)), autologging_call_args, user_call_args
    ):
        if not _is_arg_exempt_from_validation(
            autologging_integration,
            function_name,
            user_call_arg,
            argument_index=index,
        ):
            _validate(autologging_call_arg, user_call_arg)

    # Similar validation logic found in _validate, unraveling the dictionary of arguments to exclude
    # checks for any validation exempt keyword arguments.
    _assert_autologging_input_kwargs_are_superset(autologging_call_kwargs, user_call_kwargs)
    for key in autologging_call_kwargs.keys():
        if not _is_arg_exempt_from_validation(
            autologging_integration,
            function_name,
            user_call_kwargs.get(key, None),
            argument_name=key,
        ):
            _validate(
                autologging_call_kwargs[key],
                user_call_kwargs.get(key, None),
            )
