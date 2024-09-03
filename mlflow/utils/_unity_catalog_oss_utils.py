from mlflow.entities.model_registry import ModelVersion, RegisteredModel, RegisteredModelSearch, ModelVersionSearch
from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    ModelVersionInfo,
    ModelVersionStatus,
    RegisteredModelInfo,
)
import re
from mlflow.exceptions import MlflowException

_STRING_TO_STATUS = {k: ModelVersionStatus.Value(k) for k in ModelVersionStatus.keys()}
_STATUS_TO_STRING = {value: key for key, value in _STRING_TO_STATUS.items()}


def registered_model_from_uc_oss_proto(uc_oss_proto: RegisteredModelInfo) -> RegisteredModel:
    return RegisteredModel(
        name=f"{uc_oss_proto.catalog_name}.{uc_oss_proto.schema_name}.{uc_oss_proto.name}",
        creation_timestamp=uc_oss_proto.created_at,
        last_updated_timestamp=uc_oss_proto.updated_at,
        description=uc_oss_proto.comment,
    )


def model_version_from_uc_oss_proto(uc_oss_proto: ModelVersionInfo) -> ModelVersion:
    return ModelVersion(
        name=f"{uc_oss_proto.catalog_name}.{uc_oss_proto.schema_name}.{uc_oss_proto.model_name}",
        version=uc_oss_proto.version,
        creation_timestamp=uc_oss_proto.created_at,
        last_updated_timestamp=uc_oss_proto.updated_at,
        description=uc_oss_proto.comment,
        source=uc_oss_proto.source,
        run_id=uc_oss_proto.run_id,
        status=uc_oss_model_version_status_to_string(uc_oss_proto.status),
    )

def registered_model_search_from_uc_oss_proto(uc_oss_proto: RegisteredModelInfo) -> RegisteredModelSearch:
    return RegisteredModelSearch(
        name=f"{uc_oss_proto.catalog_name}.{uc_oss_proto.schema_name}.{uc_oss_proto.name}",
        creation_timestamp=uc_oss_proto.creation_timestamp,
        last_updated_timestamp=uc_oss_proto.last_updated_timestamp,
        description=uc_oss_proto.description,
    )

def model_version_search_from_uc_oss_proto(uc_oss_proto: ModelVersionInfo) -> ModelVersionSearch:
    return ModelVersionSearch(
        name=uc_oss_proto.name,
        version=uc_oss_proto.version,
        creation_timestamp=uc_oss_proto.creation_timestamp,
        last_updated_timestamp=uc_oss_proto.last_updated_timestamp,
        description=uc_oss_proto.description,
        user_id=uc_oss_proto.user_id,
        source=uc_oss_proto.source,
        run_id=uc_oss_proto.run_id,
        status=uc_oss_model_version_status_to_string(uc_oss_proto.status)
    )


def uc_oss_model_version_status_to_string(status):
    return _STATUS_TO_STRING[status]

# filter_pattern = re.compile(r"^name\s*=\\s*'([^']+)'")
filter_pattern = re.compile(r"^name\s*=\s*'([^']+)'")

def parse_model_name(filter):
    trimmed_filter = filter.strip()
    match = filter_pattern.match(trimmed_filter)
    if match:
        model_name_str = match.group(1)
    elif trimmed_filter == "":
        raise MlflowException(
            "Missing filter: please specify a filter parameter in the format `name = 'model_name'`."
        )
    else:
        raise MlflowException(
            f"Unsupported filter query : `{trimmed_filter}`. Please specify your filter parameter in " +
            "the format `name = 'model_name'`."
        )
    parts = model_name_str.split('.')
    if len(parts) != 3:
        raise MlflowException("Full name must have three parts separated by '.'")
    catalog, schema, model = parts
    if not model or not catalog or not schema:
        raise MlflowException(
            "Bad model name: please specify all three levels of the model in the form `catalog_name.schema_name.model_name`",
        )
    return f"{catalog}.{schema}.{model}"

