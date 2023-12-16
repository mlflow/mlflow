import graphene
import mlflow
from mlflow.server.graphql import graphql_schema_extensions
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict


class MlflowViewType(graphene.Enum):
    ACTIVE_ONLY = 'ACTIVE_ONLY'
    DELETED_ONLY = 'DELETED_ONLY'
    ALL = 'ALL'


class MlflowLogInputsResponse(graphene.ObjectType):
    dummy = graphene.Boolean(description='Dummy field required because GraphQL does not support empty types.')


class MlflowLogModelResponse(graphene.ObjectType):
    dummy = graphene.Boolean(description='Dummy field required because GraphQL does not support empty types.')


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


class MlflowSearchExperimentsResponse(graphene.ObjectType):
    experiments = graphene.List(graphene.NonNull(graphql_schema_extensions.ExtendedExperiment))
    next_page_token = graphene.String()


class MlflowGetExperimentByNameResponse(graphene.ObjectType):
    experiment = graphene.Field(graphql_schema_extensions.ExtendedExperiment)


class MlflowDatasetInput(graphene.InputObjectType):
    name = graphene.String()
    digest = graphene.String()
    source_type = graphene.String()
    source = graphene.String()
    schema = graphene.String()
    profile = graphene.String()


class MlflowInputTagInput(graphene.InputObjectType):
    key = graphene.String()
    value = graphene.String()


class MlflowDatasetInputInput(graphene.InputObjectType):
    tags = graphene.List(graphene.NonNull(MlflowInputTagInput))
    dataset = graphene.InputField(MlflowDatasetInput)


class MlflowLogInputsInput(graphene.InputObjectType):
    run_id = graphene.String()
    datasets = graphene.List(graphene.NonNull(MlflowDatasetInputInput))


class MlflowLogModelInput(graphene.InputObjectType):
    run_id = graphene.String()
    model_json = graphene.String()


class MlflowSearchExperimentsInput(graphene.InputObjectType):
    max_results = LongString()
    page_token = graphene.String()
    filter = graphene.String()
    order_by = graphene.List(graphene.String)
    view_type = graphene.Field(MlflowViewType)


class MlflowGetExperimentByNameInput(graphene.InputObjectType):
    experiment_name = graphene.String()


class QueryType(graphene.ObjectType):
    mlflow_get_experiment_by_name = graphene.Field(MlflowGetExperimentByNameResponse, input=MlflowGetExperimentByNameInput())

    def resolve_mlflow_get_experiment_by_name(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.GetExperimentByName()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.get_experiment_by_name_impl(request_message)


class MutationType(graphene.ObjectType):
    mlflow_search_experiments = graphene.Field(MlflowSearchExperimentsResponse, input=MlflowSearchExperimentsInput())
    mlflow_log_model = graphene.Field(MlflowLogModelResponse, input=MlflowLogModelInput())
    mlflow_log_inputs = graphene.Field(MlflowLogInputsResponse, input=MlflowLogInputsInput())

    def resolve_mlflow_search_experiments(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.SearchExperiments()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.search_experiments_impl(request_message)

    def resolve_mlflow_log_model(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.LogModel()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.log_model_impl(request_message)

    def resolve_mlflow_log_inputs(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.LogInputs()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.log_inputs_impl(request_message)
