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
        + "where suffix is a model version, stage, or the string '%s'." % _MODELS_URI_SUFFIX_LATEST
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
    Returns (name, version, stage). Since a models:/ URI can only have one of
    {version, stage, 'latest'}, it will return
        - (name, version, None) to look for a specific version,
        - (name, None, stage) to look for the latest version of a stage,
        - (name, None, None) to look for the latest of all versions.
    """
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "models":
        raise MlflowException(_improper_model_uri_msg(uri))

    path = parsed.path
    if not path.startswith("/") or len(path) <= 1:
        raise MlflowException(_improper_model_uri_msg(uri))
    parts = path[1:].split("/")

    if len(parts) != 2 or parts[0].strip() == "":
        raise MlflowException(_improper_model_uri_msg(uri))

    if parts[1].isdigit():
        # The suffix is a specific version, e.g. "models:/AdsModel1/123"
        return parts[0], int(parts[1]), None
    elif parts[1].lower() == _MODELS_URI_SUFFIX_LATEST.lower():
        # The suffix is the 'latest' string (case insensitive), e.g. "models:/AdsModel1/latest"
        return parts[0], None, None
    else:
        # The suffix is a specific stage (case insensitive), e.g. "models:/AdsModel1/Production"
        return parts[0], None, parts[1]


def get_model_name_and_version(client, models_uri):
    (model_name, model_version, model_stage) = _parse_model_uri(models_uri)
    if model_version is not None:
        return model_name, str(model_version)
    return model_name, str(_get_latest_model_version(client, model_name, model_stage))
