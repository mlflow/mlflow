from mlflow.entities import Metric
import mlflow
from mlflow.client import MlflowClient
import time
import os
from azureml.mlflow._store.tracking.store import AzureMLRestStore

if __name__ == "__main__":
    os.environ[
        "MLFLOW_TRACKING_TOKEN"
    ] = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6Ii1LSTNROW5OUjdiUm9meG1lWm9YcWJIWkdldyIsImtpZCI6Ii1LSTNROW5OUjdiUm9meG1lWm9YcWJIWkdldyJ9.eyJhdWQiOiJodHRwczovL21hbmFnZW1lbnQuY29yZS53aW5kb3dzLm5ldC8iLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC83MmY5ODhiZi04NmYxLTQxYWYtOTFhYi0yZDdjZDAxMWRiNDcvIiwiaWF0IjoxNjk1MjQ2MTEzLCJuYmYiOjE2OTUyNDYxMTMsImV4cCI6MTY5NTI1MTYxNywiX2NsYWltX25hbWVzIjp7Imdyb3VwcyI6InNyYzEifSwiX2NsYWltX3NvdXJjZXMiOnsic3JjMSI6eyJlbmRwb2ludCI6Imh0dHBzOi8vZ3JhcGgud2luZG93cy5uZXQvNzJmOTg4YmYtODZmMS00MWFmLTkxYWItMmQ3Y2QwMTFkYjQ3L3VzZXJzL2JmZjM1NWY4LTFiMmYtNDExOC1hODI5LWJlMzJiNDFhMTk5Zi9nZXRNZW1iZXJPYmplY3RzIn19LCJhY3IiOiIxIiwiYWlvIjoiQVlRQWUvOFVBQUFBNW1LMWtJKzJpeFFCUUVnbjZXUjlYNmdVZkJMK01ncmdtanVjVG1uTi8zRThtcFhkRDJNQkl6RUFGcXhFck85azBRSTFaSW92TmVrTGtLOU5SNEdyWDIxeXZNMGgzVW5UZlVWeWRkOFN1V0ZUQU92bXZDVUVXK25mc1dWWU9taHRJOE9YRjR1eDUwSktOMUdEVmxFb0xaamhVZ0F0aE8vbkI4ZEdjbjZPUUxVPSIsImFtciI6WyJyc2EiLCJtZmEiXSwiYXBwaWQiOiIwNGIwNzc5NS04ZGRiLTQ2MWEtYmJlZS0wMmY5ZTFiZjdiNDYiLCJhcHBpZGFjciI6IjAiLCJkZXZpY2VpZCI6ImJmMzUyZDFhLWRkMDAtNDVkNC1hZTNkLWJkZTRlZTAxNjUyNCIsImZhbWlseV9uYW1lIjoiU3VtYW50IiwiZ2l2ZW5fbmFtZSI6IlNhZ2FyIiwiaXBhZGRyIjoiNTAuNDcuMjM4LjI0NiIsIm5hbWUiOiJTYWdhciBTdW1hbnQiLCJvaWQiOiJiZmYzNTVmOC0xYjJmLTQxMTgtYTgyOS1iZTMyYjQxYTE5OWYiLCJvbnByZW1fc2lkIjoiUy0xLTUtMjEtMjEyNzUyMTE4NC0xNjA0MDEyOTIwLTE4ODc5Mjc1MjctODEzMjg2NCIsInB1aWQiOiIxMDAzN0ZGRTgwREVCMTZGIiwicmgiOiIwLkFSb0F2NGo1Y3ZHR3IwR1JxeTE4MEJIYlIwWklmM2tBdXRkUHVrUGF3ZmoyTUJNYUFNNC4iLCJzY3AiOiJ1c2VyX2ltcGVyc29uYXRpb24iLCJzdWIiOiJFaG1fbG9faUhjUkRhdlVQOHlKTWN4ZURNLVVJbDU5T3pISVJ0SHJkM2RJIiwidGlkIjoiNzJmOTg4YmYtODZmMS00MWFmLTkxYWItMmQ3Y2QwMTFkYjQ3IiwidW5pcXVlX25hbWUiOiJzYXN1bUBtaWNyb3NvZnQuY29tIiwidXBuIjoic2FzdW1AbWljcm9zb2Z0LmNvbSIsInV0aSI6IlZHUk1qaGNuTkVpall3MGhxZ0pGQUEiLCJ2ZXIiOiIxLjAiLCJ3aWRzIjpbImI3OWZiZjRkLTNlZjktNDY4OS04MTQzLTc2YjE5NGU4NTUwOSJdLCJ4bXNfY2MiOlsiQ1AxIl0sInhtc190Y2R0IjoxMjg5MjQxNTQ3fQ.rwyeEyAVlptxtrjNLhLycPA6V9szq-yf_jSsvdUExSIpIy15dMn7_8nfPhqpcvVGrXKFZCoBCcJJJCmac2eVau_9cTsW0CrGbWIGQiPy737ceEQrnnm28RJyDIk3jqYdEWw1ESsZzTOZU7oUD78aCLEFuYjcRvufOwVVGuKVR4RHKhtH69E-um02cgE5s0hsjnafxIi46lgrSovBgC1yJ6kEe3hYZrjqTbx6uvSJiZY43AgzWYIrUsX83tr_N5EPNIzTnksw59YXbhEBfAq9GuAdo-SOzKMeM_lyn2alDDWuXcFRlAA-NFQrzs2vHXQ6rnaDuP0Xm9HNep-ncCuTFg"
    mlflow.set_tracking_uri(
        "http://localhost:56171/mlflow/v2.0/subscriptions/381b38e9-9840-4719-a5a0-61d9585e1e91/resourceGroups/sasum_centraluseuap_rg/providers/Microsoft.MachineLearningServices/workspaces/sasum-int-ws-2"
    )

    experiment_name = "my-mlflow-experiment-running-locally-from-AMLMlflowClient"
    mlflow.set_experiment(experiment_name)

    mlflow_client = MlflowClient()
    store = mlflow_client._tracking_client.store
    run = mlflow.start_run()
    run_id = run.info.run_id

    # Log batch of metrics
    mlflow_client.log_batch(
        run_id,
        metrics=[Metric(key="sample_list", value=val, timestamp=int(time.time() * 1000), step=0) for val in range(100)],
        synchronous=False,
    )

    store = mlflow_client._tracking_client.store
    aml_store = store if isinstance(store, AzureMLRestStore) else store.aml_store
    aml_store.await_run_data(run_id=run_id)

    # AML Mlflowclient await_run_data
    # run_data_await_operation = mlflow_client.await_run_data(run_id=run_id)
    # run_data_await_operation.await_completion()

    # pure mlflow logging metrics in fluent style
    mlflow_client = MlflowClient(
        "http://localhost:56171/mlflow/v2.0/subscriptions/381b38e9-9840-4719-a5a0-61d9585e1e91/resourceGroups/sasum_centraluseuap_rg/providers/Microsoft.MachineLearningServices/workspaces/sasum-int-ws-2"
    )
    with mlflow.start_run() as run:
        mlflow_client.log_metric("from_mlflow_client", 1, synchronous=False)
        mlflow_client.log_metrics({"c": 1, "d": 2}, synchronous=False)
        mlflow.log_metric("sasum-metric-3", time.time(), synchronous=False)
        mlflow.log_metrics({"a": 1, "b": 2}, synchronous=False)

    print("hello")
