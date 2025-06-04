import logging
import time
from typing import Optional

from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace_info_v2 import TraceInfoV2
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.processor.base_mlflow import BaseMlflowSpanProcessor
from mlflow.tracing.utils import get_otel_attribute
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID

_logger = logging.getLogger(__name__)


class MlflowV2SpanProcessor(BaseMlflowSpanProcessor):
    """
    Defines custom hooks to be executed when a span is started or ended (before exporting).

    This processor is used for exporting traces to MLflow Tracking Server
    using the V2 trace schema and API.
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        tracking_uri: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ):
        super().__init__(span_exporter, experiment_id)
        self._client = TracingClient(tracking_uri)

        # We issue a warning when a trace is created under the default experiment.
        # We only want to issue it once, and typically it can be achieved by using
        # warnings.warn() with filterwarnings setting. However, the de-duplication does
        # not work in notebooks (https://github.com/ipython/ipython/issues/11207),
        # so we instead keep track of the warning issuance state manually.
        self._issued_default_exp_warning = False

    def _start_trace(self, span: OTelSpan) -> TraceInfoV2:
        # If the user started trace/span with fixed start time, this attribute is set
        start_time_ns = get_otel_attribute(span, SpanAttributeKey.START_TIME_NS)

        experiment_id = self._get_experiment_id_for_trace(span)
        if experiment_id == DEFAULT_EXPERIMENT_ID and not self._issued_default_exp_warning:
            _logger.warning(
                "Creating a trace within the default experiment with id "
                f"'{DEFAULT_EXPERIMENT_ID}'. It is strongly recommended to not use "
                "the default experiment to log traces due to ambiguous search results and "
                "probable performance issues over time due to directory table listing performance "
                "degradation with high volumes of directories within a specific path. "
                "To avoid performance and disambiguation issues, set the experiment for "
                "your environment using `mlflow.set_experiment()` API."
            )
            self._issued_default_exp_warning = True

        trace_info = self._client.start_trace(
            experiment_id=experiment_id,
            # TODO: This timestamp is not accurate because it is not adjusted to exclude the
            #   latency of the backend API call. We do this adjustment for span start time
            #   above, but can't do it for trace start time until the backend API supports
            #   updating the trace start time.
            timestamp_ms=(start_time_ns or span.start_time) // 1_000_000,  # ns to ms
            request_metadata=self._get_basic_trace_metadata(),
            tags=self._get_basic_trace_tags(span),
        )

        self._trace_manager.register_trace(span.context.trace_id, trace_info.to_v3())

        # NB: This is a workaround to exclude the latency of backend StartTrace API call (within
        #   _create_trace_info()) from the execution time of the span. The API call takes ~1 sec
        #   and significantly skews the span duration.
        if not start_time_ns:
            span._start_time = time.time_ns()

        return trace_info
