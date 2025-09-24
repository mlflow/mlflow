from mlflow.entities.trace_location import TraceLocation, TraceLocationType, UCSchemaLocation
from mlflow.protos import databricks_service_pb2 as pb


def uc_schema_location_to_proto(uc_schema_location: UCSchemaLocation) -> pb.UCSchemaLocation:
    return pb.UCSchemaLocation(
        catalog_name=uc_schema_location.catalog_name,
        schema_name=uc_schema_location.schema_name,
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
