import re

from mlflow.entities.model_registry import (
    ModelVersion,
    ModelVersionSearch,
    RegisteredModel,
    RegisteredModelSearch,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    ModelVersionInfo,
    ModelVersionStatus,
    RegisteredModelInfo,
)

_STRING_TO_STATUS = {k: ModelVersionStatus.Value(k) for k in ModelVersionStatus.keys()}
_STATUS_TO_STRING = {value: key for key, value in _STRING_TO_STATUS.items()}


def get_registered_model_from_uc_oss_proto(uc_oss_proto: RegisteredModelInfo) -> RegisteredModel:
    return RegisteredModel(
        name=f"{uc_oss_proto.catalog_name}.{uc_oss_proto.schema_name}.{uc_oss_proto.name}",
        creation_timestamp=uc_oss_proto.created_at,
        last_updated_timestamp=uc_oss_proto.updated_at,
        description=uc_oss_proto.comment,
    )


def get_model_version_from_uc_oss_proto(uc_oss_proto: ModelVersionInfo) -> ModelVersion:
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


def get_registered_model_search_from_uc_oss_proto(
    uc_oss_proto: RegisteredModelInfo,
) -> RegisteredModelSearch:
    return RegisteredModelSearch(
        name=f"{uc_oss_proto.catalog_name}.{uc_oss_proto.schema_name}.{uc_oss_proto.name}",
        creation_timestamp=uc_oss_proto.created_at,
        last_updated_timestamp=uc_oss_proto.updated_at,
        description=uc_oss_proto.comment,
    )


def get_model_version_search_from_uc_oss_proto(
    uc_oss_proto: ModelVersionInfo,
) -> ModelVersionSearch:
    return ModelVersionSearch(
        name=f"{uc_oss_proto.catalog_name}.{uc_oss_proto.schema_name}.{uc_oss_proto.model_name}",
        version=uc_oss_proto.version,
        creation_timestamp=uc_oss_proto.created_at,
        last_updated_timestamp=uc_oss_proto.updated_at,
        description=uc_oss_proto.comment,
        source=uc_oss_proto.source,
        run_id=uc_oss_proto.run_id,
        status=uc_oss_model_version_status_to_string(uc_oss_proto.status),
    )


def uc_oss_model_version_status_to_string(status):
    return _STATUS_TO_STRING[status]


filter_pattern = re.compile(r"^name\s*=\s*'([^']+)'")


def parse_model_name(filter):
    trimmed_filter = filter.strip()
    if match := filter_pattern.match(trimmed_filter):
        model_name_str = match.group(1)
    elif trimmed_filter == "":
        raise MlflowException(
            "Missing filter: please specify a filter parameter in the format `name = 'model_name'`."
        )
    else:
        raise MlflowException(
            f"Unsupported filter query : `{trimmed_filter}`."
            + " Please specify your filter parameter in "
            + "the format `name = 'model_name'`."
        )
    parts = model_name_str.split(".")
    if len(parts) != 3 or not all(parts):
        raise MlflowException(
            "Bad model name: please specify all three levels of the model in the"
            "form `catalog_name.schema_name.model_name`"
        )
    catalog, schema, model = parts
    return f"{catalog}.{schema}.{model}"
