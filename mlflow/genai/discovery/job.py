from mlflow.client import MlflowClient
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS
from mlflow.exceptions import MlflowException
from mlflow.genai.discovery.pipeline import discover_issues
from mlflow.server.jobs import job
from mlflow.store.tracking import MAX_TRACE_LINKS_PER_REQUEST
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.providers import _CORE_PROVIDER_ENV_VARS


def _fetch_provider_credentials(
    store: SqlAlchemyStore, provider: str, secret_id: str
) -> dict[str, str]:
    """
    Retrieve and extract provider credentials from secret.

    Args:
        provider: The provider name (e.g., "openai", "anthropic").
        secret_id: The ID of the secret containing the credentials.
    """
    env_var_config = _CORE_PROVIDER_ENV_VARS.get(provider.lower())
    if env_var_config is None:
        raise MlflowException(
            f"Unknown provider '{provider}'. "
            f"Supported providers: {', '.join(_CORE_PROVIDER_ENV_VARS.keys())}. "
            "To use other providers, create an AI Gateway endpoint instead."
        )

    secret_value = store._get_decrypted_secret(secret_id)
    credentials = {}
    if isinstance(env_var_config, dict):
        for secret_key, env_var_name in env_var_config.items():
            if value := secret_value.get(secret_key):
                credentials[env_var_name] = value
    else:
        if api_key := secret_value.get("api_key"):
            credentials[env_var_config] = api_key

    return credentials


@job(name="invoke_issue_detection", max_workers=MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS.get())
def invoke_issue_detection_job(
    experiment_id: str,
    trace_ids: list[str],
    categories: list[str],
    run_id: str,
    model: str | None = None,
):
    """
    Job function to run issue detection on traces.

    Note: Provider credentials should be passed via environment variables set by submit_job().
    """
    client = MlflowClient()
    try:
        # Cannot link more than MAX_TRACE_LINKS_PER_REQUEST traces to a run in a single request
        for i in range(0, len(trace_ids), MAX_TRACE_LINKS_PER_REQUEST):
            batch = trace_ids[i : i + MAX_TRACE_LINKS_PER_REQUEST]
            client.link_traces_to_run(batch, run_id)
        traces = client._tracing_client.batch_get_traces(trace_ids)
        result = discover_issues(
            experiment_id=experiment_id,
            traces=traces,
            model=model,
            run_id=run_id,
            categories=categories,
        )
        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED))
        client.set_tag(run_id, "total_cost_usd", result.total_cost_usd)
        return {
            "summary": result.summary,
            "issues": len(result.issues),
            "total_traces_analyzed": result.total_traces_analyzed,
            "total_cost_usd": result.total_cost_usd,
        }
    except Exception:
        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FAILED))
        raise
