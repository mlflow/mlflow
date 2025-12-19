import logging

from google.protobuf.duration_pb2 import Duration
from google.protobuf.timestamp_pb2 import Timestamp

from mlflow.entities import Assessment, Span, Trace, TraceData, TraceInfo
from mlflow.entities.trace_info_v2 import _truncate_request_metadata, _truncate_tags
from mlflow.entities.trace_location import (
    InferenceTableLocation,
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
    UCSchemaLocation,
)
from mlflow.protos import assessments_pb2
from mlflow.protos import databricks_tracing_pb2 as pb
from mlflow.tracing.utils import (
    construct_trace_id_v4,
    parse_trace_id_v4,
)

_logger = logging.getLogger(__name__)


def uc_schema_location_to_proto(uc_schema_location: UCSchemaLocation) -> pb.UCSchemaLocation:
    return pb.UCSchemaLocation(
        catalog_name=uc_schema_location.catalog_name,
        schema_name=uc_schema_location.schema_name,
        otel_spans_table_name=uc_schema_location._otel_spans_table_name,
        otel_logs_table_name=uc_schema_location._otel_logs_table_name,
    )


def uc_schema_location_from_proto(proto: pb.UCSchemaLocation) -> UCSchemaLocation:
    location = UCSchemaLocation(catalog_name=proto.catalog_name, schema_name=proto.schema_name)

    if proto.HasField("otel_spans_table_name"):
        location._otel_spans_table_name = proto.otel_spans_table_name
    if proto.HasField("otel_logs_table_name"):
        location._otel_logs_table_name = proto.otel_logs_table_name
    return location


def inference_table_location_to_proto(
    inference_table_location: InferenceTableLocation,
) -> pb.InferenceTableLocation:
    return pb.InferenceTableLocation(full_table_name=inference_table_location.full_table_name)


def mlflow_experiment_location_to_proto(
    mlflow_experiment_location: MlflowExperimentLocation,
) -> pb.MlflowExperimentLocation:
    return pb.MlflowExperimentLocation(experiment_id=mlflow_experiment_location.experiment_id)


def trace_location_to_proto(trace_location: TraceLocation) -> pb.TraceLocation:
    if trace_location.type == TraceLocationType.UC_SCHEMA:
        return pb.TraceLocation(
            type=pb.TraceLocation.TraceLocationType.UC_SCHEMA,
            uc_schema=uc_schema_location_to_proto(trace_location.uc_schema),
        )
    elif trace_location.type == TraceLocationType.MLFLOW_EXPERIMENT:
        return pb.TraceLocation(
            type=pb.TraceLocation.TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=mlflow_experiment_location_to_proto(trace_location.mlflow_experiment),
        )
    elif trace_location.type == TraceLocationType.INFERENCE_TABLE:
        return pb.TraceLocation(
            type=pb.TraceLocation.TraceLocationType.INFERENCE_TABLE,
            inference_table=inference_table_location_to_proto(trace_location.inference_table),
        )
    else:
        raise ValueError(f"Unsupported trace location type: {trace_location.type}")


def trace_location_type_from_proto(proto: pb.TraceLocation.TraceLocationType) -> TraceLocationType:
    return TraceLocationType(pb.TraceLocation.TraceLocationType.Name(proto))


def trace_location_from_proto(proto: pb.TraceLocation) -> TraceLocation:
    type_ = trace_location_type_from_proto(proto.type)

    if proto.WhichOneof("identifier") == "uc_schema":
        return TraceLocation(
            type=type_,
            uc_schema=uc_schema_location_from_proto(proto.uc_schema),
        )
    elif proto.WhichOneof("identifier") == "mlflow_experiment":
        return TraceLocation(
            type=type_,
            mlflow_experiment=MlflowExperimentLocation.from_proto(proto.mlflow_experiment),
        )
    elif proto.WhichOneof("identifier") == "inference_table":
        return TraceLocation(
            type=type_,
            inference_table=InferenceTableLocation.from_proto(proto.inference_table),
        )
    else:
        return TraceLocation(TraceLocationType.TRACE_LOCATION_TYPE_UNSPECIFIED)


def trace_info_to_v4_proto(trace_info: TraceInfo) -> pb.TraceInfo:
    request_time = Timestamp()
    request_time.FromMilliseconds(trace_info.request_time)
    execution_duration = None
    if trace_info.execution_duration is not None:
        execution_duration = Duration()
        execution_duration.FromMilliseconds(trace_info.execution_duration)

    if trace_info.trace_location.uc_schema:
        _, trace_id = parse_trace_id_v4(trace_info.trace_id)
    else:
        trace_id = trace_info.trace_id

    return pb.TraceInfo(
        trace_id=trace_id,
        client_request_id=trace_info.client_request_id,
        trace_location=trace_location_to_proto(trace_info.trace_location),
        request_preview=trace_info.request_preview,
        response_preview=trace_info.response_preview,
        request_time=request_time,
        execution_duration=execution_duration,
        state=pb.TraceInfo.State.Value(trace_info.state),
        trace_metadata=_truncate_request_metadata(trace_info.trace_metadata),
        tags=_truncate_tags(trace_info.tags),
        assessments=[assessment_to_proto(a) for a in trace_info.assessments],
    )


def trace_to_proto(trace: Trace) -> pb.Trace:
    return pb.Trace(
        trace_info=trace.info.to_proto(),
        spans=[span.to_otel_proto() for span in trace.data.spans],
    )


def trace_from_proto(proto: pb.Trace, location_id: str) -> Trace:
    return Trace(
        info=TraceInfo.from_proto(proto.trace_info),
        data=TraceData(spans=[Span.from_otel_proto(span, location_id) for span in proto.spans]),
    )


def assessment_to_proto(assessment: Assessment) -> pb.Assessment:
    assessment_proto = pb.Assessment()
    assessment_proto.assessment_name = assessment.name
    location, trace_id = parse_trace_id_v4(assessment.trace_id)
    if location:
        catalog, schema = location.split(".")
        assessment_proto.trace_location.CopyFrom(
            pb.TraceLocation(
                type=pb.TraceLocation.TraceLocationType.UC_SCHEMA,
                uc_schema=pb.UCSchemaLocation(catalog_name=catalog, schema_name=schema),
            )
        )
    assessment_proto.trace_id = trace_id

    assessment_proto.source.CopyFrom(assessment.source.to_proto())

    # Convert time in milliseconds to protobuf Timestamp
    assessment_proto.create_time.FromMilliseconds(assessment.create_time_ms)
    assessment_proto.last_update_time.FromMilliseconds(assessment.last_update_time_ms)

    if assessment.span_id is not None:
        assessment_proto.span_id = assessment.span_id
    if assessment.rationale is not None:
        assessment_proto.rationale = assessment.rationale
    if assessment.assessment_id is not None:
        assessment_proto.assessment_id = assessment.assessment_id

    if assessment.expectation is not None:
        assessment_proto.expectation.CopyFrom(assessment.expectation.to_proto())
    elif assessment.feedback is not None:
        assessment_proto.feedback.CopyFrom(assessment.feedback.to_proto())

    if assessment.metadata:
        for key, value in assessment.metadata.items():
            assessment_proto.metadata[key] = str(value)
    if assessment.overrides:
        assessment_proto.overrides = assessment.overrides
    if assessment.valid is not None:
        assessment_proto.valid = assessment.valid

    return assessment_proto


def get_trace_id_from_assessment_proto(proto: pb.Assessment | assessments_pb2.Assessment) -> str:
    if "trace_location" in proto.DESCRIPTOR.fields_by_name and proto.trace_location.HasField(
        "uc_schema"
    ):
        trace_location = proto.trace_location
        return construct_trace_id_v4(
            f"{trace_location.uc_schema.catalog_name}.{trace_location.uc_schema.schema_name}",
            proto.trace_id,
        )
    else:
        return proto.trace_id
