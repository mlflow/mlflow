from mlflow.client import MlflowClient
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS
from mlflow.genai.discovery.pipeline import discover_issues
from mlflow.server.jobs import job


@job(name="invoke_issue_detection", max_workers=MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS.get())
def invoke_issue_detection_job(
    experiment_id: str,
    trace_ids: list[str],
    categories: list[str],
    provider: str,
    model: str,
    run_id: str,
):
    client = MlflowClient()
    try:
        traces = client._tracing_client.batch_get_traces(trace_ids)
        # can not link more than 100 traces to a run in a single request
        for i in range(0, len(trace_ids), 100):
            batch = trace_ids[i : i + 100]
            client.link_traces_to_run(batch, run_id)
        result = discover_issues(
            experiment_id=experiment_id,
            traces=traces,
            categories=categories,
            model=f"{provider}:/{model}",
            run_id=run_id,
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
