from google.protobuf.duration_pb2 import Duration
from google.protobuf.timestamp_pb2 import Timestamp

from mlflow.entities import Span, Trace, TraceData, TraceInfo, TraceState
from mlflow.entities.trace_info_v2 import _truncate_request_metadata, _truncate_tags
from mlflow.entities.trace_location import TraceLocation, TraceLocationType, UCSchemaLocation
from mlflow.protos import databricks_tracing_pb2 as pb
from mlflow.tracing.utils import parse_trace_id_v4


def uc_schema_location_to_proto(uc_schema_location: UCSchemaLocation) -> pb.UCSchemaLocation:
    return pb.UCSchemaLocation(
        catalog_name=uc_schema_location.catalog_name,
        schema_name=uc_schema_location.schema_name,
        otel_spans_table_name=uc_schema_location.otel_spans_table_name,
        otel_logs_table_name=uc_schema_location.otel_logs_table_name,
    )


def trace_location_to_proto(trace_location: TraceLocation) -> pb.TraceLocation:
    if trace_location.type == TraceLocationType.UC_SCHEMA:
        return pb.TraceLocation(
            type=pb.TraceLocation.TraceLocationType.UC_SCHEMA,
            uc_schema=uc_schema_location_to_proto(trace_location.uc_schema),
        )
    elif trace_location.type == TraceLocationType.MLFLOW_EXPERIMENT:
        return pb.TraceLocation(
            type=pb.TraceLocation.TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=pb.TraceLocation.MlflowExperimentLocation(
                experiment_id=trace_location.mlflow_experiment.experiment_id
            ),
        )
    elif trace_location.type == TraceLocationType.INFERENCE_TABLE:
        return pb.TraceLocation(
            type=pb.TraceLocation.TraceLocationType.INFERENCE_TABLE,
            inference_table=pb.TraceLocation.InferenceTableLocation(
                full_table_name=trace_location.inference_table.full_table_name
            ),
        )
    else:
        raise ValueError(f"Unsupported trace location type: {trace_location.type}")


def trace_location_type_from_proto(proto: pb.TraceLocation.TraceLocationType) -> TraceLocationType:
    return TraceLocationType(pb.TraceLocation.TraceLocationType.Name(proto))


def trace_state_to_proto(trace_state: TraceState) -> pb.TraceInfo.State:
    return pb.TraceInfo.State.Value(trace_state)


def trace_info_to_proto(trace_info: TraceInfo) -> pb.TraceInfo:
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
        state=trace_state_to_proto(trace_info.state),
        trace_metadata=_truncate_request_metadata(trace_info.trace_metadata),
        tags=_truncate_tags(trace_info.tags),
        # TODO: update once assessment proto is updated
        assessments=[a.to_proto() for a in trace_info.assessments],
    )


def trace_to_proto(trace: Trace) -> pb.Trace:
    return pb.Trace(
        trace_info=trace_info_to_proto(trace.info),
        spans=[span.to_otel_proto() for span in trace.data.spans],
    )


def trace_from_proto(proto: pb.Trace) -> Trace:
    return Trace(
        info=TraceInfo.from_proto(proto.trace_info),
        data=TraceData(spans=[Span.from_otel_proto(span) for span in proto.spans]),
    )
