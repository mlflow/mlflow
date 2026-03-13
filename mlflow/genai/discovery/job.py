from mlflow.client import MlflowClient
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS
from mlflow.exceptions import MlflowException
from mlflow.genai.discovery.pipeline import discover_issues
from mlflow.server.jobs import job
from mlflow.store.tracking.utils.secrets import get_decrypted_secret

# Mapping of provider names to their environment variable names for API keys
# Providers that don't use API keys are not included in this mapping
_PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_API_KEY",  # Azure OpenAI
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_API_KEY",
    "ai21labs": "AI21_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "togetherai": "TOGETHERAI_API_KEY",
    "mosaicml": "MOSAICML_API_KEY",
    "palm": "PALM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "bedrock": {
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_session_token": "AWS_SESSION_TOKEN",  # Optional
    },
    "amazon-bedrock": {  # Alias for bedrock
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_session_token": "AWS_SESSION_TOKEN",
    },
}


def _fetch_provider_credentials(provider: str, secret_id: str) -> dict[str, str]:
    """
    Retrieve and extract provider credentials from secret.

    Args:
        provider: The provider name (e.g., "openai", "anthropic").
        secret_id: The ID of the secret containing the credentials.
    """
    env_var_config = _PROVIDER_ENV_VARS.get(provider.lower())
    if env_var_config is None:
        raise MlflowException(
            f"Unknown provider '{provider}'. "
            f"Supported providers: {', '.join(_PROVIDER_ENV_VARS.keys())}"
        )

    secret_value = get_decrypted_secret(secret_id)
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
    provider: str | None = None,
    model: str | None = None,
    endpoint_name: str | None = None,
):
    """
    Job function to run issue detection on traces.

    Note: Provider credentials should be passed via environment variables set by submit_job().
    """
    client = MlflowClient()
    try:
        # can not link more than 100 traces to a run in a single request
        for i in range(0, len(trace_ids), 100):
            batch = trace_ids[i : i + 100]
            client.link_traces_to_run(batch, run_id)
        traces = client._tracing_client.batch_get_traces(trace_ids)
        model_name = endpoint_name or f"{provider}:/{model}"
        result = discover_issues(
            experiment_id=experiment_id,
            traces=traces,
            model=model_name,
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
