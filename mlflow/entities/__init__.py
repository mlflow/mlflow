"""
The ``mlflow.entities`` module defines entities returned by the MLflow
`REST API <../rest-api.html>`_.
"""

from mlflow.entities.assessment import (
    Assessment,
    AssessmentError,
    AssessmentSource,
    AssessmentSourceType,
    Expectation,
    Feedback,
)
from mlflow.entities.dataset import Dataset
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.dataset_record import DatasetRecord
from mlflow.entities.dataset_record_source import DatasetRecordSource, DatasetRecordSourceType
from mlflow.entities.dataset_summary import _DatasetSummary
from mlflow.entities.document import Document
from mlflow.entities.entity_type import EntityAssociationType
from mlflow.entities.experiment import Experiment
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.entities.file_info import FileInfo
from mlflow.entities.gateway_endpoint import (
    GatewayEndpoint,
    GatewayEndpointBinding,
    GatewayEndpointModelMapping,
    GatewayEndpointTag,
    GatewayModelDefinition,
    GatewayResourceType,
)
from mlflow.entities.gateway_secrets import GatewaySecretInfo
from mlflow.entities.input_tag import InputTag
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.logged_model import LoggedModel
from mlflow.entities.logged_model_input import LoggedModelInput
from mlflow.entities.logged_model_output import LoggedModelOutput
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.metric import Metric
from mlflow.entities.model_registry import Prompt
from mlflow.entities.param import Param
from mlflow.entities.run import Run
from mlflow.entities.run_data import RunData
from mlflow.entities.run_info import RunInfo
from mlflow.entities.run_inputs import RunInputs
from mlflow.entities.run_outputs import RunOutputs
from mlflow.entities.run_status import RunStatus
from mlflow.entities.run_tag import RunTag
from mlflow.entities.scorer import ScorerVersion
from mlflow.entities.source_type import SourceType
from mlflow.entities.span import LiveSpan, NoOpSpan, Span, SpanType
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatus, SpanStatusCode
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import (
    InferenceTableLocation,
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
    UCSchemaLocation,
)
from mlflow.entities.trace_state import TraceState
from mlflow.entities.view_type import ViewType
from mlflow.entities.webhook import (
    Webhook,
    WebhookEvent,
    WebhookStatus,
    WebhookTestResult,
)

__all__ = [
    "Experiment",
    "ExperimentTag",
    "FileInfo",
    "Metric",
    "Param",
    "Prompt",
    "Run",
    "RunData",
    "RunInfo",
    "RunStatus",
    "RunTag",
    "ScorerVersion",
    "SourceType",
    "ViewType",
    "LifecycleStage",
    "Dataset",
    "InputTag",
    "DatasetInput",
    "RunInputs",
    "RunOutputs",
    "Span",
    "LiveSpan",
    "NoOpSpan",
    "SpanEvent",
    "SpanStatus",
    "SpanType",
    "Trace",
    "TraceData",
    "TraceInfo",
    "TraceLocation",
    "TraceLocationType",
    "MlflowExperimentLocation",
    "InferenceTableLocation",
    "UCSchemaLocation",
    "TraceState",
    "SpanStatusCode",
    "_DatasetSummary",
    "LoggedModel",
    "LoggedModelInput",
    "LoggedModelOutput",
    "LoggedModelStatus",
    "LoggedModelTag",
    "LoggedModelParameter",
    "Document",
    "Assessment",
    "AssessmentError",
    "AssessmentSource",
    "AssessmentSourceType",
    "Expectation",
    "Feedback",
    # Note: EvaluationDataset is intentionally excluded from __all__ to prevent
    # circular import issues during plugin registration. It can still be imported
    # explicitly via: from mlflow.entities import EvaluationDataset
    "DatasetRecord",
    "DatasetRecordSource",
    "DatasetRecordSourceType",
    "EntityAssociationType",
    "GatewayEndpoint",
    "GatewayEndpointBinding",
    "GatewayEndpointModelMapping",
    "GatewayEndpointTag",
    "GatewayModelDefinition",
    "GatewayResourceType",
    "GatewaySecretInfo",
    "Webhook",
    "WebhookEvent",
    "WebhookStatus",
    "WebhookTestResult",
]


def __getattr__(name):
    """Lazy loading for EvaluationDataset to avoid circular imports."""
    if name == "EvaluationDataset":
        try:
            from mlflow.entities.evaluation_dataset import EvaluationDataset

            return EvaluationDataset
        except ImportError:
            # EvaluationDataset requires mlflow.data which may not be available
            # in minimal installations like mlflow-tracing
            raise AttributeError(
                "EvaluationDataset is not available. It requires the mlflow.data module "
                "which is not included in this installation."
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
