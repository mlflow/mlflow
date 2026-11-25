import urllib.parse
from pathlib import Path
from typing import NamedTuple

import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import (
    get_databricks_profile_uri_from_artifact_uri,
    is_databricks_uri,
    is_models_uri,
)

_MODELS_URI_SUFFIX_LATEST = "latest"


def is_using_databricks_registry(uri, registry_uri: str | None = None):
    profile_uri = (
        get_databricks_profile_uri_from_artifact_uri(uri)
        or registry_uri
        or mlflow.get_registry_uri()
    )
    return is_databricks_uri(profile_uri)


def _improper_model_uri_msg(uri, scheme: str = "models"):
    if scheme not in ("models", "prompts"):
        raise ValueError(f"Unsupported scheme for model/prompt URI: {scheme!r}")
    entity_type = "Models" if scheme == "models" else "Prompts"
    return (
        f"Not a proper {scheme}:/ URI: {uri}. "
        + f"{entity_type} URIs must be of the form '{scheme}:/name/suffix' "
        + f"or '{scheme}:/name@alias' where suffix is a version"
        + (f", stage, or the string {_MODELS_URI_SUFFIX_LATEST!r}" if scheme == "models" else "")
        + f" and where alias is a registered {scheme[:-1]} alias. "
        + "Only one of suffix or alias can be defined at a time."
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
    model_id: str | None = None
    name: str | None = None
    version: str | None = None
    stage: str | None = None
    alias: str | None = None


def _parse_model_uri(uri, scheme: str = "models") -> ParsedModelUri:
    """
    Returns a ParsedModelUri tuple. Since a models:/ or prompts:/ URI can only have one of
    {version, stage, 'latest', alias}, it will return
        - (id, None, None, None) to look for a specific model by ID,
        - (name, version, None, None) to look for a specific version,
        - (name, None, stage, None) to look for the latest version of a stage,
        - (name, None, None, None) to look for the latest of all versions.
        - (name, None, None, alias) to look for a registered model alias.

    Args:
        uri: The URI to parse (e.g., "models:/name/version" or "prompts:/name@alias")
        scheme: The expected URI scheme (default: "models", can be "prompts")
    """
    parsed = urllib.parse.urlparse(uri, allow_fragments=False)
    if parsed.scheme != scheme:
        raise MlflowException(_improper_model_uri_msg(uri, scheme))
    path = parsed.path
    if not path.startswith("/") or len(path) <= 1:
        raise MlflowException(_improper_model_uri_msg(uri, scheme))

    parts = path.lstrip("/").split("/")
    if len(parts) > 2 or parts[0].strip() == "":
        raise MlflowException(_improper_model_uri_msg(uri, scheme))

    if len(parts) == 2:
        name, suffix = parts
        if suffix.strip() == "":
            raise MlflowException(_improper_model_uri_msg(uri, scheme))
        # The URI is in the suffix format
        if suffix.isdigit():
            # The suffix is a specific version, e.g. "models:/AdsModel1/123"
            return ParsedModelUri(name=name, version=suffix)
        elif suffix.lower() == _MODELS_URI_SUFFIX_LATEST.lower() and scheme == "models":
            # The suffix is the 'latest' string (case insensitive), e.g. "models:/AdsModel1/latest"
            # Only supported for models, not prompts
            return ParsedModelUri(name=name)
        elif scheme == "models":
            # The suffix is a specific stage (case insensitive), e.g. "models:/AdsModel1/Production"
            # Only supported for models, not prompts
            return ParsedModelUri(name=name, stage=suffix)
        else:
            # For prompts, only version numbers are supported, not stages or 'latest'
            raise MlflowException(_improper_model_uri_msg(uri, scheme))
    elif "@" in path:
        # The URI is an alias URI, e.g. "models:/AdsModel1@Champion"
        alias_parts = parts[0].rsplit("@", 1)
        if len(alias_parts) != 2 or alias_parts[1].strip() == "":
            raise MlflowException(_improper_model_uri_msg(uri, scheme))
        return ParsedModelUri(name=alias_parts[0], alias=alias_parts[1])
    else:
        # The URI is of the form "models:/<model_id>"
        return ParsedModelUri(parts[0])


def _resolve_model_uri_and_id(model_uri: str | Path) -> tuple[str, str | None]:
    """Pin an alias to its version so the artifacts and model ID refer to the same model."""
    uri = str(model_uri)
    if not is_models_uri(uri):
        return uri, None

    parsed = _parse_model_uri(uri)
    if parsed.version is None and parsed.alias is None:
        return uri, parsed.model_id

    registry_uri = get_databricks_profile_uri_from_artifact_uri(uri)
    client = mlflow.tracking.MlflowClient(registry_uri=registry_uri)
    if parsed.alias is None:
        return uri, client.get_model_version(parsed.name, parsed.version).model_id

    version = client.get_model_version_by_alias(parsed.name, parsed.alias)
    url = urllib.parse.urlparse(uri, allow_fragments=False)
    return url._replace(path=f"/{parsed.name}/{version.version}").geturl(), version.model_id


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
