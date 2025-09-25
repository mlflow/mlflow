from mlflow.entities.trace_location import (
    InferenceTableLocation,
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
    UCSchemaLocation,
)
from mlflow.protos import databricks_tracing_pb2 as pb


def uc_schema_location_to_proto(uc_schema_location: UCSchemaLocation) -> pb.UCSchemaLocation:
    return pb.UCSchemaLocation(
        catalog_name=uc_schema_location.catalog_name,
        schema_name=uc_schema_location.schema_name,
    )


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
