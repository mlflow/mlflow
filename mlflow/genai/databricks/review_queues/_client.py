from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.tracking import get_tracking_uri
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import http_request, verify_rest_response
from mlflow.utils.uri import is_databricks_uri

_API_PREFIX = "/api/2.0/managed-evals"


def call(
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return call_api(method, f"{_API_PREFIX}/{path}", json=json, params=params)


def call_api(
    method: str,
    endpoint: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Calls any Databricks REST endpoint, given its full path, e.g. `/api/2.0/permissions/...`.
    tracking_uri = get_tracking_uri()
    if not is_databricks_uri(tracking_uri):
        raise MlflowException.invalid_parameter_value(
            "`mlflow.genai.databricks.review_queues` requires a Databricks tracking URI; "
            f"got {tracking_uri!r}. Use `mlflow.genai.review_queues` with other tracking servers."
        )
    kwargs = {}
    if json is not None:
        kwargs["json"] = json
    if params := {k: v for k, v in (params or {}).items() if v is not None}:
        kwargs["params"] = params
    response = http_request(
        host_creds=get_databricks_host_creds(tracking_uri),
        endpoint=endpoint,
        method=method,
        **kwargs,
    )
    return verify_rest_response(response, endpoint).json()
