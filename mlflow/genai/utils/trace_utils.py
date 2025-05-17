import logging
from typing import Any, Callable

from opentelemetry.trace import NoOpTracer

from mlflow.tracing.provider import trace_disabled

_logger = logging.getLogger(__name__)


def is_model_traced(predict_fn: Callable, sample_input: Any):
    """
    Check if the predict function is being traced without logging to the database.

    The check is done in the following steps so it doesn't have any side effects:
        1. Disable tracing.
        2. Patch the NoOpTracer.start_span method to count the number of times it is called.
            NoOpTracer is used when tracing is disabled.
        3. Call the predict function with the sample input.
        4. Restore the original NoOpTracer.start_span method and re-enable tracing.

    WARNING: This function is not thread-safe. We do not provide support for running
        `mlflow.genai.evaluate` in multi-threaded environments.`

    Args:
        predict_fn: The predict function to be evaluated.
        sample_input: A sample input to the model.

    Returns:
        True if the function is being traced, False otherwise.
    """

    @trace_disabled
    def _check():
        original = NoOpTracer.start_span
        counter = 0

        def _patched_start_span(self, *args, **kwargs):
            nonlocal counter
            counter += 1
            return original(*args, **kwargs)

        NoOpTracer.start_span = _patched_start_span
        try:
            predict_fn(**sample_input)
        except Exception as e:
            _logger.debug(
                "Tried to make a single prediction to check if the model is traced, "
                f"but got an error: {e}. Assuming the model is not traced."
            )

        NoOpTracer.start_span = original
        return counter > 0

    return _check()
