# GENERATED FILE. PLEASE DON'T MODIFY.
# Run python3 ./dev/proto_to_graphql/code_generator.py to regenerate.
import graphene

import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict


class MlflowRunStatus(graphene.Enum):
    RUNNING = "RUNNING"
    SCHEDULED = "SCHEDULED"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


class MlflowDataset(graphene.ObjectType):
    name = graphene.String()
    digest = graphene.String()
    source_type = graphene.String()
    source = graphene.String()
    schema = graphene.String()
    profile = graphene.String()


class MlflowInputTag(graphene.ObjectType):
    key = graphene.String()
    value = graphene.String()


class MlflowDatasetInput(graphene.ObjectType):
    tags = graphene.List(graphene.NonNull(MlflowInputTag))
    dataset = graphene.Field(MlflowDataset)


class MlflowRunInputs(graphene.ObjectType):
    dataset_inputs = graphene.List(graphene.NonNull(MlflowDatasetInput))


class MlflowRunTag(graphene.ObjectType):
    key = graphene.String()
    value = graphene.String()


class MlflowParam(graphene.ObjectType):
    key = graphene.String()
    value = graphene.String()


class MlflowMetric(graphene.ObjectType):
    key = graphene.String()
    value = graphene.Float()
    timestamp = LongString()
    step = LongString()


class MlflowRunData(graphene.ObjectType):
    metrics = graphene.List(graphene.NonNull(MlflowMetric))
    params = graphene.List(graphene.NonNull(MlflowParam))
    tags = graphene.List(graphene.NonNull(MlflowRunTag))


class MlflowRunInfo(graphene.ObjectType):
    run_id = graphene.String()
    run_uuid = graphene.String()
    run_name = graphene.String()
    experiment_id = graphene.String()
    user_id = graphene.String()
    status = graphene.Field(MlflowRunStatus)
    start_time = LongString()
    end_time = LongString()
    artifact_uri = graphene.String()
    lifecycle_stage = graphene.String()


class MlflowRun(graphene.ObjectType):
    info = graphene.Field(MlflowRunInfo)
    data = graphene.Field(MlflowRunData)
    inputs = graphene.Field(MlflowRunInputs)


class MlflowGetRunResponse(graphene.ObjectType):
    run = graphene.Field(MlflowRun)


class MlflowGetRunInput(graphene.InputObjectType):
    run_id = graphene.String()
    run_uuid = graphene.String()


class QueryType(graphene.ObjectType):
    mlflow_get_run = graphene.Field(MlflowGetRunResponse, input=MlflowGetRunInput())

    def resolve_mlflow_get_run(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.GetRun()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.get_run_impl(request_message)


class MutationType(graphene.ObjectType):
    pass
