import urllib.parse
from pathlib import Path
from typing import NamedTuple, Optional, Union

import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import (
    get_databricks_profile_uri_from_artifact_uri,
    is_databricks_uri,
    is_models_uri,
)

_MODELS_URI_SUFFIX_LATEST = "latest"


def is_using_databricks_registry(uri):
    profile_uri = get_databricks_profile_uri_from_artifact_uri(uri) or mlflow.get_registry_uri()
    return is_databricks_uri(profile_uri)


def _improper_model_uri_msg(uri):
    return (
        f"Not a proper models:/ URI: {uri}. "
        + "Models URIs must be of the form 'models:/model_name/suffix' "
        + "or 'models:/model_name@alias' where suffix is a model version, stage, "
        + f"or the string {_MODELS_URI_SUFFIX_LATEST!r} and where alias is a registered model "
        + "alias. Only one of suffix or alias can be defined at a time."
    )


def _get_latest_model_version(client, name, stage):
    """
    Returns the latest version of the stage if stage is not None. Otherwise return the latest of all
    versions.
    """
    latest = client.get_latest_versions(name, None if stage is None else [stage])
    if len(latest) == 0:
        stage_str = "" if stage is None else f" and stage '{stage}'"
        raise MlflowException(f"No versions of model with name '{name}'{stage_str} found")
    return max(int(x.version) for x in latest)


class ParsedModelUri(NamedTuple):
    model_id: Optional[str] = None
    name: Optional[str] = None
    version: Optional[str] = None
    stage: Optional[str] = None
    alias: Optional[str] = None


def _parse_model_uri(uri) -> ParsedModelUri:
    """
    Returns a ParsedModelUri tuple. Since a models:/ URI can only have one of
    {version, stage, 'latest', alias}, it will return
        - (id, None, None, None) to look for a specific model by ID,
        - (name, version, None, None) to look for a specific version,
        - (name, None, stage, None) to look for the latest version of a stage,
        - (name, None, None, None) to look for the latest of all versions.
        - (name, None, None, alias) to look for a registered model alias.
    """
    parsed = urllib.parse.urlparse(uri, allow_fragments=False)
    if parsed.scheme != "models":
        raise MlflowException(_improper_model_uri_msg(uri))
    path = parsed.path
    if not path.startswith("/") or len(path) <= 1:
        raise MlflowException(_improper_model_uri_msg(uri))

    parts = path.lstrip("/").split("/")
    if len(parts) > 2 or parts[0].strip() == "":
        raise MlflowException(_improper_model_uri_msg(uri))

    if len(parts) == 2:
        name, suffix = parts
        if suffix.strip() == "":
            raise MlflowException(_improper_model_uri_msg(uri))
        # The URI is in the suffix format
        if suffix.isdigit():
            # The suffix is a specific version, e.g. "models:/AdsModel1/123"
            return ParsedModelUri(name=name, version=suffix)
        elif suffix.lower() == _MODELS_URI_SUFFIX_LATEST.lower():
            # The suffix is the 'latest' string (case insensitive), e.g. "models:/AdsModel1/latest"
            return ParsedModelUri(name=name)
        else:
            # The suffix is a specific stage (case insensitive), e.g. "models:/AdsModel1/Production"
            return ParsedModelUri(name=name, stage=suffix)
    elif "@" in path:
        # The URI is an alias URI, e.g. "models:/AdsModel1@Champion"
        alias_parts = parts[0].rsplit("@", 1)
        if len(alias_parts) != 2 or alias_parts[1].strip() == "":
            raise MlflowException(_improper_model_uri_msg(uri))
        return ParsedModelUri(name=alias_parts[0], alias=alias_parts[1])
    else:
        # The URI is of the form "models:/<model_id>"
        return ParsedModelUri(parts[0])


def _parse_model_id_if_present(possible_model_uri: Union[str, Path]) -> Optional[str]:
    """
    Parses the model ID from the given string. If the string represents a UC model URI, we get the
    model version to extract the model ID. If the string is not a models:/ URI, returns None.

    Args:
        possible_model_uri: The string that may be a models:/ URI.

    Returns:
        The model ID if the string is a models:/ URI, otherwise None.
    """
    uri = str(possible_model_uri)
    if is_models_uri(uri):
        parsed_model_uri = _parse_model_uri(uri)
        if parsed_model_uri.model_id is not None:
            return parsed_model_uri.model_id
        elif parsed_model_uri.name is not None and parsed_model_uri.version is not None:
            client = mlflow.tracking.MlflowClient()
            return client.get_model_version(
                parsed_model_uri.name, parsed_model_uri.version
            ).model_id
    return None


def get_model_name_and_version(client, models_uri):
    (model_id, model_name, model_version, model_stage, model_alias) = _parse_model_uri(models_uri)
    if model_id is not None:
        return (model_id,)
    if model_version is not None:
        return model_name, model_version

    # NB: Call get_model_version_by_alias of registry client directly to bypass prompt check
    if isinstance(client, mlflow.MlflowClient):
        client = client._get_registry_client()

    if model_alias is not None:
        mv = client.get_model_version_by_alias(model_name, model_alias)
        return model_name, mv.version
    return model_name, str(_get_latest_model_version(client, model_name, model_stage))
