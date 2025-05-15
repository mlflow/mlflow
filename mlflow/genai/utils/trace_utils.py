import logging
from typing import Any, Callable

from opentelemetry.trace import NoOpTracer

import mlflow
from mlflow.genai.utils.data_validation import check_model_prediction

_logger = logging.getLogger(__name__)


def convert_predict_fn(predict_fn: Callable, sample_input: Any) -> Callable:
    """
    Check the predict_fn is callable and add trace decorator if it is not already traced.
    """
    with NoOpTracerPatcher() as counter:
        check_model_prediction(predict_fn, sample_input)

    if counter.count == 0:
        predict_fn = mlflow.trace(predict_fn)

    # Wrap the prediction function to unwrap the inputs dictionary into keyword arguments.
    return lambda request: predict_fn(**request)


class NoOpTracerPatcher:
    """
    A context manager to count the number of times NoOpTracer's start_span is called.

    The check is done in the following steps so it doesn't have any side effects:
    1. Disable tracing.
    2. Patch the NoOpTracer.start_span method to count the number of times it is called.
        NoOpTracer is used when tracing is disabled.
    3. Call the predict function with the sample input.
    4. Restore the original NoOpTracer.start_span method and re-enable tracing.


    WARNING: This function is not thread-safe. We do not provide support for running
        `mlflow.genai.evaluate` in multi-threaded environments.`
    """

    def __init__(self):
        self.count = 0

    def __enter__(self):
        self.original = NoOpTracer.start_span

        def _patched_start_span(_self, *args, **kwargs):
            self.count += 1
            return self.original(_self, *args, **kwargs)

        NoOpTracer.start_span = _patched_start_span
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        NoOpTracer.start_span = self.original
