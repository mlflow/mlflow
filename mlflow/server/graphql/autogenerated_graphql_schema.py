# GENERATED FILE. PLEASE DON'T MODIFY.
# Run python3 ./dev/proto_to_graphql/code_generator.py to regenerate.
import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.server.graphql.graphql_errors import ApiError
from mlflow.utils.proto_json_utils import parse_dict


class MlflowDeploymentJobConnectionState(graphene.Enum):
    DEPLOYMENT_JOB_CONNECTION_STATE_UNSPECIFIED = 1
    NOT_SET_UP = 2
    CONNECTED = 3
    NOT_FOUND = 4
    REQUIRED_PARAMETERS_CHANGED = 5


class MlflowModelVersionDeploymentJobStateDeploymentJobRunState(graphene.Enum):
    DEPLOYMENT_JOB_RUN_STATE_UNSPECIFIED = 1
    NO_VALID_DEPLOYMENT_JOB_FOUND = 2
    RUNNING = 3
    SUCCEEDED = 4
    FAILED = 5
    PENDING = 6
    APPROVAL = 7


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


class MlflowViewType(graphene.Enum):
    ACTIVE_ONLY = 1
    DELETED_ONLY = 2
    ALL = 3


class MlflowModelVersionDeploymentJobState(graphene.ObjectType):
    job_id = graphene.String()
    run_id = graphene.String()
    job_state = graphene.Field(MlflowDeploymentJobConnectionState)
    run_state = graphene.Field(MlflowModelVersionDeploymentJobStateDeploymentJobRunState)
    current_task_name = graphene.String()


class MlflowModelMetric(graphene.ObjectType):
    key = graphene.String()
    value = graphene.Float()
    timestamp = LongString()
    step = LongString()
    dataset_name = graphene.String()
    dataset_digest = graphene.String()
    model_id = graphene.String()
    run_id = graphene.String()


class MlflowModelParam(graphene.ObjectType):
    name = graphene.String()
    value = graphene.String()


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
    model_id = graphene.String()
    model_params = graphene.List(graphene.NonNull(MlflowModelParam))
    model_metrics = graphene.List(graphene.NonNull(MlflowModelMetric))
    deployment_job_state = graphene.Field(MlflowModelVersionDeploymentJobState)


class MlflowSearchModelVersionsResponse(graphene.ObjectType):
    model_versions = graphene.List(graphene.NonNull(MlflowModelVersion))
    next_page_token = graphene.String()
    apiError = graphene.Field(ApiError)


class MlflowDatasetSummary(graphene.ObjectType):
    experiment_id = graphene.String()
    name = graphene.String()
    digest = graphene.String()
    context = graphene.String()


class MlflowSearchDatasetsResponse(graphene.ObjectType):
    dataset_summaries = graphene.List(graphene.NonNull(MlflowDatasetSummary))
    apiError = graphene.Field(ApiError)


class MlflowMetricWithRunId(graphene.ObjectType):
    key = graphene.String()
    value = graphene.Float()
    timestamp = LongString()
    step = LongString()
    run_id = graphene.String()


class MlflowGetMetricHistoryBulkIntervalResponse(graphene.ObjectType):
    metrics = graphene.List(graphene.NonNull(MlflowMetricWithRunId))
    apiError = graphene.Field(ApiError)


class MlflowFileInfo(graphene.ObjectType):
    path = graphene.String()
    is_dir = graphene.Boolean()
    file_size = LongString()


class MlflowListArtifactsResponse(graphene.ObjectType):
    root_uri = graphene.String()
    files = graphene.List(graphene.NonNull(MlflowFileInfo))
    next_page_token = graphene.String()
    apiError = graphene.Field(ApiError)


class MlflowModelOutput(graphene.ObjectType):
    model_id = graphene.String()
    step = LongString()


class MlflowRunOutputs(graphene.ObjectType):
    model_outputs = graphene.List(graphene.NonNull(MlflowModelOutput))


class MlflowModelInput(graphene.ObjectType):
    model_id = graphene.String()


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
    model_inputs = graphene.List(graphene.NonNull(MlflowModelInput))


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
    dataset_name = graphene.String()
    dataset_digest = graphene.String()
    model_id = graphene.String()
    run_id = graphene.String()


class MlflowRunData(graphene.ObjectType):
    metrics = graphene.List(graphene.NonNull('mlflow.server.graphql.graphql_schema_extensions.MlflowMetricExtension'))
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
    outputs = graphene.Field(MlflowRunOutputs)


class MlflowSearchRunsResponse(graphene.ObjectType):
    runs = graphene.List(graphene.NonNull('mlflow.server.graphql.graphql_schema_extensions.MlflowRunExtension'))
    next_page_token = graphene.String()
    apiError = graphene.Field(ApiError)


class MlflowGetRunResponse(graphene.ObjectType):
    run = graphene.Field('mlflow.server.graphql.graphql_schema_extensions.MlflowRunExtension')
    apiError = graphene.Field(ApiError)


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
    apiError = graphene.Field(ApiError)


class MlflowSearchModelVersionsInput(graphene.InputObjectType):
    filter = graphene.String()
    max_results = LongString()
    order_by = graphene.List(graphene.String)
    page_token = graphene.String()


class MlflowSearchDatasetsInput(graphene.InputObjectType):
    experiment_ids = graphene.List(graphene.String)


class MlflowGetMetricHistoryBulkIntervalInput(graphene.InputObjectType):
    run_ids = graphene.List(graphene.String)
    metric_key = graphene.String()
    start_step = graphene.Int()
    end_step = graphene.Int()
    max_results = graphene.Int()


class MlflowListArtifactsInput(graphene.InputObjectType):
    run_id = graphene.String()
    run_uuid = graphene.String()
    path = graphene.String()
    page_token = graphene.String()


class MlflowSearchRunsInput(graphene.InputObjectType):
    experiment_ids = graphene.List(graphene.String)
    filter = graphene.String()
    run_view_type = graphene.Field(MlflowViewType)
    max_results = graphene.Int()
    order_by = graphene.List(graphene.String)
    page_token = graphene.String()


class MlflowGetRunInput(graphene.InputObjectType):
    run_id = graphene.String()
    run_uuid = graphene.String()


class MlflowGetExperimentInput(graphene.InputObjectType):
    experiment_id = graphene.String()


class QueryType(graphene.ObjectType):
    mlflow_get_experiment = graphene.Field(MlflowGetExperimentResponse, input=MlflowGetExperimentInput())
    mlflow_get_metric_history_bulk_interval = graphene.Field(MlflowGetMetricHistoryBulkIntervalResponse, input=MlflowGetMetricHistoryBulkIntervalInput())
    mlflow_get_run = graphene.Field(MlflowGetRunResponse, input=MlflowGetRunInput())
    mlflow_list_artifacts = graphene.Field(MlflowListArtifactsResponse, input=MlflowListArtifactsInput())
    mlflow_search_model_versions = graphene.Field(MlflowSearchModelVersionsResponse, input=MlflowSearchModelVersionsInput())

    def resolve_mlflow_get_experiment(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.GetExperiment()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.get_experiment_impl(request_message)

    def resolve_mlflow_get_metric_history_bulk_interval(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.GetMetricHistoryBulkInterval()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.get_metric_history_bulk_interval_impl(request_message)

    def resolve_mlflow_get_run(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.GetRun()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.get_run_impl(request_message)

    def resolve_mlflow_list_artifacts(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.ListArtifacts()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.list_artifacts_impl(request_message)

    def resolve_mlflow_search_model_versions(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.model_registry_pb2.SearchModelVersions()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.search_model_versions_impl(request_message)


class MutationType(graphene.ObjectType):
    mlflow_search_datasets = graphene.Field(MlflowSearchDatasetsResponse, input=MlflowSearchDatasetsInput())
    mlflow_search_runs = graphene.Field(MlflowSearchRunsResponse, input=MlflowSearchRunsInput())

    def resolve_mlflow_search_datasets(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.SearchDatasets()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.search_datasets_impl(request_message)

    def resolve_mlflow_search_runs(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.SearchRuns()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.search_runs_impl(request_message)
