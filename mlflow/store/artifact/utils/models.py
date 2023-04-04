import urllib.parse

import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri, is_databricks_uri

_MODELS_URI_SUFFIX_LATEST = "latest"


def is_using_databricks_registry(uri):
    profile_uri = get_databricks_profile_uri_from_artifact_uri(uri) or mlflow.get_registry_uri()
    return is_databricks_uri(profile_uri)


def _improper_model_uri_msg(uri):
    return (
        "Not a proper models:/ URI: %s. " % uri
        + "Models URIs must be of the form 'models:/<model_name>/suffix' "
        + "or 'models:/<model_name>@alias' where suffix is a model version, stage, "
        + "or the string '%s' and where alias is a registered model alias. "
        % _MODELS_URI_SUFFIX_LATEST
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


def _parse_model_uri(uri):
    """
    Returns (name, version, stage, alias). Since a models:/ URI can only have one of
    {version, stage, 'latest', alias}, it will return
        - (name, version, None, None) to look for a specific version,
        - (name, None, stage, None) to look for the latest version of a stage,
        - (name, None, None, None) to look for the latest of all versions.
        - (name, None, None, alias) to look for a registered model alias.
    """
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "models":
        raise MlflowException(_improper_model_uri_msg(uri))

    path = parsed.path
    if not path.startswith("/") or len(path) <= 1:
        raise MlflowException(_improper_model_uri_msg(uri))
    stage_parts = path[1:].split("/")
    alias_parts = path[1:].split("@")

    if (
        stage_parts[0].strip() == ""
        or alias_parts[0].strip() == ""  # no model name
        or (len(stage_parts) == 1 and len(alias_parts) != 2)  # improper alias URI
        or (len(stage_parts) == 2 and len(alias_parts) != 1)  # improper stage URI
        or len(stage_parts) > 2
        or len(alias_parts) > 2  # improper URI
    ):
        raise MlflowException(_improper_model_uri_msg(uri))

    if len(alias_parts) == 2:
        # The URI is an alias URI, e.g. "models:/AdsModel1@Champion"
        return alias_parts[0], None, None, alias_parts[1]
    if stage_parts[1].isdigit():
        # The suffix is a specific version, e.g. "models:/AdsModel1/123"
        return stage_parts[0], int(stage_parts[1]), None, None
    elif stage_parts[1].lower() == _MODELS_URI_SUFFIX_LATEST.lower():
        # The suffix is the 'latest' string (case insensitive), e.g. "models:/AdsModel1/latest"
        return stage_parts[0], None, None, None
    else:
        # The suffix is a specific stage (case insensitive), e.g. "models:/AdsModel1/Production"
        return stage_parts[0], None, stage_parts[1], None


def get_model_name_and_version(client, models_uri):
    (model_name, model_version, model_stage, model_alias) = _parse_model_uri(models_uri)
    if model_version is not None:
        return model_name, str(model_version)
    if model_alias is not None:
        return model_name, client.get_model_version_by_alias(model_name, model_alias).version
    return model_name, str(_get_latest_model_version(client, model_name, model_stage))
