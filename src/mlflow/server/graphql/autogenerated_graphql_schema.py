# GENERATED FILE. PLEASE DON'T MODIFY.
# Run python3 ./dev/proto_to_graphql/code_generator.py to regenerate.
import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict


class MlflowModelVersionStatus(graphene.Enum):
    PENDING_REGISTRATION = 1
    FAILED_REGISTRATION = 2
    READY = 3


class MlflowRunStatus(graphene.Enum):
    RUNNING = 1
    SCHEDULED = 2
    FINISHED = 3
    FAILED = 4
    KILLED = 5


class MlflowModelVersionTag(graphene.ObjectType):
    key = graphene.String()
    value = graphene.String()


class MlflowModelVersion(graphene.ObjectType):
    name = graphene.String()
    version = graphene.String()
    creation_timestamp = LongString()
    last_updated_timestamp = LongString()
    user_id = graphene.String()
    current_stage = graphene.String()
    description = graphene.String()
    source = graphene.String()
    run_id = graphene.String()
    status = graphene.Field(MlflowModelVersionStatus)
    status_message = graphene.String()
    tags = graphene.List(graphene.NonNull(MlflowModelVersionTag))
    run_link = graphene.String()
    aliases = graphene.List(graphene.String)


class MlflowSearchModelVersionsResponse(graphene.ObjectType):
    model_versions = graphene.List(graphene.NonNull(MlflowModelVersion))
    next_page_token = graphene.String()


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
    run = graphene.Field('mlflow.server.graphql.graphql_schema_extensions.MlflowRunExtension')


class MlflowExperimentTag(graphene.ObjectType):
    key = graphene.String()
    value = graphene.String()


class MlflowExperiment(graphene.ObjectType):
    experiment_id = graphene.String()
    name = graphene.String()
    artifact_location = graphene.String()
    lifecycle_stage = graphene.String()
    last_update_time = LongString()
    creation_time = LongString()
    tags = graphene.List(graphene.NonNull(MlflowExperimentTag))


class MlflowGetExperimentResponse(graphene.ObjectType):
    experiment = graphene.Field(MlflowExperiment)


class MlflowSearchModelVersionsInput(graphene.InputObjectType):
    filter = graphene.String()
    max_results = LongString()
    order_by = graphene.List(graphene.String)
    page_token = graphene.String()


class MlflowGetRunInput(graphene.InputObjectType):
    run_id = graphene.String()
    run_uuid = graphene.String()


class MlflowGetExperimentInput(graphene.InputObjectType):
    experiment_id = graphene.String()


class QueryType(graphene.ObjectType):
    mlflow_get_experiment = graphene.Field(MlflowGetExperimentResponse, input=MlflowGetExperimentInput())
    mlflow_get_run = graphene.Field(MlflowGetRunResponse, input=MlflowGetRunInput())
    mlflow_search_model_versions = graphene.Field(MlflowSearchModelVersionsResponse, input=MlflowSearchModelVersionsInput())

    def resolve_mlflow_get_experiment(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.GetExperiment()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.get_experiment_impl(request_message)

    def resolve_mlflow_get_run(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.GetRun()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.get_run_impl(request_message)

    def resolve_mlflow_search_model_versions(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.model_registry_pb2.SearchModelVersions()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.search_model_versions_impl(request_message)


class MutationType(graphene.ObjectType):
    pass
