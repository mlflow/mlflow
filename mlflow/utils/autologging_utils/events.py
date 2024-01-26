import warnings

from mlflow.utils.autologging_utils import _logger


class AutologgingEventLogger:
    """
    Provides instrumentation hooks for important autologging lifecycle events, including:

        - Calls to `mlflow.autolog()` APIs
        - Calls to patched APIs with associated termination states
          ("success" and "failure due to error")
        - Calls to original / underlying APIs made by patched function code with
          associated termination states ("success" and "failure due to error")

    Default implementations are included for each of these hooks, which emit corresponding
    DEBUG-level logging statements. Developers can provide their own hook implementations
    by subclassing `AutologgingEventLogger` and calling the static
    `AutologgingEventLogger.set_logger()` method to supply a new event logger instance.

    Callers fetch the configured logger via `AutologgingEventLogger.get_logger()`
    and invoke one or more hooks (e.g., `AutologgingEventLogger.get_logger().log_autolog_called()`).
    """

    _event_logger = None

    @staticmethod
    def get_logger():
        """Fetches the configured `AutologgingEventLogger` instance for logging.

        Returns:
            The instance of `AutologgingEventLogger` specified via `set_logger`
            (if configured) or the default implementation of `AutologgingEventLogger`
            (if a logger was not configured via `set_logger`).

        """
        return AutologgingEventLogger._event_logger or AutologgingEventLogger()

    @staticmethod
    def set_logger(logger):
        """Configures the `AutologgingEventLogger` instance for logging. This instance
        is exposed via `AutologgingEventLogger.get_logger()` and callers use it to invoke
        logging hooks (e.g., AutologgingEventLogger.get_logger().log_autolog_called()).

        Args:
            logger: The instance of `AutologgingEventLogger` to use when invoking logging hooks.

        """
        AutologgingEventLogger._event_logger = logger

    def log_autolog_called(self, integration, call_args, call_kwargs):
        """Called when the `autolog()` method for an autologging integration
        is invoked (e.g., when a user invokes `mlflow.sklearn.autolog()`)

        Args:
            integration: The autologging integration for which `autolog()` was called.
            call_args: **DEPRECATED** The positional arguments passed to the `autolog()` call.
                This field is empty in MLflow > 1.13.1; all arguments are passed in
                keyword form via `call_kwargs`.
            call_kwargs: The arguments passed to the `autolog()` call in keyword form.
                Any positional arguments should also be converted to keyword form
                and passed via `call_kwargs`.
        """
        if len(call_args) > 0:
            warnings.warn(
                "Received %d positional arguments via `call_args`. `call_args` is"
                " deprecated in MLflow > 1.13.1, and all arguments should be passed"
                " in keyword form via `call_kwargs`." % len(call_args),
                category=DeprecationWarning,
                stacklevel=2,
            )
        _logger.debug(
            "Called autolog() method for %s autologging with args '%s' and kwargs '%s'",
            integration,
            call_args,
            call_kwargs,
        )

    def log_patch_function_start(self, session, patch_obj, function_name, call_args, call_kwargs):
        """Called upon invocation of a patched API associated with an autologging integration
        (e.g., `sklearn.linear_model.LogisticRegression.fit()`).

        Args:
            session: The `AutologgingSession` associated with the patched API call.
            patch_obj: The object (class, module, etc) on which the patched API was called.
            function_name: The name of the patched API that was called.
            call_args: The positional arguments passed to the patched API call.
            call_kwargs: The keyword arguments passed to the patched API call.

        """
        _logger.debug(
            "Invoked patched API '%s.%s' for %s autologging with args '%s' and kwargs '%s'",
            patch_obj,
            function_name,
            session.integration,
            call_args,
            call_kwargs,
        )

    def log_patch_function_success(self, session, patch_obj, function_name, call_args, call_kwargs):
        """
        Called upon successful termination of a patched API associated with an autologging
        integration (e.g., `sklearn.linear_model.LogisticRegression.fit()`).

        Args:
            session: The `AutologgingSession` associated with the patched API call.
            patch_obj: The object (class, module, etc) on which the patched API was called.
            function_name: The name of the patched API that was called.
            call_args: The positional arguments passed to the patched API call.
            call_kwargs: The keyword arguments passed to the patched API call.
        """
        _logger.debug(
            "Patched API call '%s.%s' for %s autologging completed successfully. Patched ML"
            " API was called with args '%s' and kwargs '%s'",
            patch_obj,
            function_name,
            session.integration,
            call_args,
            call_kwargs,
        )

    def log_patch_function_error(
        self, session, patch_obj, function_name, call_args, call_kwargs, exception
    ):
        """Called when execution of a patched API associated with an autologging integration
        (e.g., `sklearn.linear_model.LogisticRegression.fit()`) terminates with an exception.

        Args:
            session: The `AutologgingSession` associated with the patched API call.
            patch_obj: The object (class, module, etc) on which the patched API was called.
            function_name: The name of the patched API that was called.
            call_args: The positional arguments passed to the patched API call.
            call_kwargs: The keyword arguments passed to the patched API call.
            exception: The exception that caused the patched API call to terminate.
        """
        _logger.debug(
            "Patched API call '%s.%s' for %s autologging threw exception. Patched API was"
            " called with args '%s' and kwargs '%s'. Exception: %s",
            patch_obj,
            function_name,
            session.integration,
            call_args,
            call_kwargs,
            exception,
        )

    def log_original_function_start(
        self, session, patch_obj, function_name, call_args, call_kwargs
    ):
        """
        Called during the execution of a patched API associated with an autologging integration
        when the original / underlying API is invoked. For example, this is called when
        a patched implementation of `sklearn.linear_model.LogisticRegression.fit()` invokes
        the original implementation of `sklearn.linear_model.LogisticRegression.fit()`.

        Args:
            session: The `AutologgingSession` associated with the patched API call.
            patch_obj: The object (class, module, etc) on which the original API was called.
            function_name: The name of the original API that was called.
            call_args: The positional arguments passed to the original API call.
            call_kwargs: The keyword arguments passed to the original API call.
        """
        _logger.debug(
            "Original function invoked during execution of patched API '%s.%s' for %s"
            " autologging. Original function was invoked with args '%s' and kwargs '%s'",
            patch_obj,
            function_name,
            session.integration,
            call_args,
            call_kwargs,
        )

    def log_original_function_success(
        self, session, patch_obj, function_name, call_args, call_kwargs
    ):
        """Called during the execution of a patched API associated with an autologging integration
        when the original / underlying API invocation terminates successfully. For example,
        when a patched implementation of `sklearn.linear_model.LogisticRegression.fit()` invokes the
        original / underlying implementation of `LogisticRegression.fit()`, then this function is
        called if the original / underlying implementation successfully completes.

        Args:
            session: The `AutologgingSession` associated with the patched API call.
            patch_obj: The object (class, module, etc) on which the original API was called.
            function_name: The name of the original API that was called.
            call_args: The positional arguments passed to the original API call.
            call_kwargs: The keyword arguments passed to the original API call.

        """
        _logger.debug(
            "Original function invocation completed successfully during execution of patched API"
            " call '%s.%s' for %s autologging. Original function was invoked with with"
            " args '%s' and kwargs '%s'",
            patch_obj,
            function_name,
            session.integration,
            call_args,
            call_kwargs,
        )

    def log_original_function_error(
        self, session, patch_obj, function_name, call_args, call_kwargs, exception
    ):
        """Called during the execution of a patched API associated with an autologging integration
        when the original / underlying API invocation terminates with an error. For example,
        when a patched implementation of `sklearn.linear_model.LogisticRegression.fit()` invokes the
        original / underlying implementation of `LogisticRegression.fit()`, then this function is
        called if the original / underlying implementation terminates with an exception.

        Args:
            session: The `AutologgingSession` associated with the patched API call.
            patch_obj: The object (class, module, etc) on which the original API was called.
            function_name: The name of the original API that was called.
            call_args: The positional arguments passed to the original API call.
            call_kwargs: The keyword arguments passed to the original API call.
            exception: The exception that caused the original API call to terminate.
        """
        _logger.debug(
            "Original function invocation threw exception during execution of patched"
            " API call '%s.%s' for %s autologging. Original function was invoked with"
            " args '%s' and kwargs '%s'. Exception: %s",
            patch_obj,
            function_name,
            session.integration,
            call_args,
            call_kwargs,
            exception,
        )
