from mlflow.entities import Metric, Param, RunTag
import mlflow
from mlflow.client import MlflowClient
import time

if __name__ == "__main__":
    mlflow.set_tracking_uri(
        "http://localhost:56171/mlflow/v1.1/subscriptions/381b38e9-9840-4719-a5a0-61d9585e1e91/resourceGroups/sasum_centraluseuap_rg/providers/Microsoft.MachineLearningServices/workspaces/sasum-int-ws-2"
    )

    experiment_name = "my-mlflow-experiment-running-locally-from-AMLMlflowClient"
    mlflow.set_experiment(experiment_name)

    mlflow_client = MlflowClient()
    store = mlflow_client._tracking_client.store
    run = mlflow.start_run()
    run_id = run.info.run_id

    mlflow_client.log_param(run_id, "single param1", str(time.time()), synchronous=False)

    mlflow_client.set_tag(run_id, "single tag 1", str(time.time()), synchronous=False)
    mlflow_client.set_tag(run_id, "single tag 2", str(time.time()), synchronous=False)

    # Log batch of metrics
    mlflow_client.log_batch(
        run_id,
        params=[Param(f"batch param{val}", value=str(time.time())) for val in range(10)],
        tags=[RunTag(f"batch tag{val}", value=str(time.time())) for val in range(10)],
        metrics=[
            Metric(key="batch metrics async", value=time.time(), timestamp=int(time.time() * 1000), step=0)
            for val in range(250)
        ],
        synchronous=False,
    )

    # Await for run data to be ingested.
    store = mlflow_client._tracking_client.store  # This would AzureMLRestStore
    run_data_await_operation = store.await_run_data(run_id=run_id)
    run_data_await_operation.await_completion()

    # Using fluent syntax
    mlflow.log_param("from-fluent-single-param-1", str(time.time()), synchronous=False)
    mlflow.set_tag("from-fluent-single-tag-1", str(time.time()), synchronous=False)
    mlflow.log_metric("from-fluent-single-metric-1", str(time.time()), synchronous=False)

    params = {}
    for val in range(10):
        params[f"from-fluent-batch-param-{val}"] = str(time.time())

    tags = {}
    for val in range(10):
        tags[f"from-fluent-batch-tag-{val}"] = str(time.time())

    mlflow.log_params(params=params, synchronous=False)
    mlflow.set_tags(tags=tags, synchronous=False)
    mlflow.log_metrics({"from-fluent-batch-metric-1": 1, "from-fluent-batch-metric-2": 2}, synchronous=False)

    print("hello")
