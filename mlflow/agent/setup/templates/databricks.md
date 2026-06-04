### Configure the Databricks workspace

You're sending traces to a Databricks workspace (`MLFLOW_TRACKING_URI={{ tracking_uri }}`).
Before instrumenting the app, make sure the workspace is reachable:

- `DATABRICKS_HOST` must be exported (e.g. `https://your-workspace.cloud.databricks.com`).
- `DATABRICKS_TOKEN` must be exported (a Databricks personal access token).
- `MLFLOW_REGISTRY_URI=databricks-uc` must be exported so the Unity Catalog
  model registry is used.

If any of those are missing, stop and ask the user. Have them `export` the
values in their shell or add to the project's existing env file. Never write
the token into a file that could be committed.

When installing MLflow in step 1, prefer `mlflow[databricks]` over plain
`mlflow` so the Databricks SDK extras are pulled in.

Pin the active experiment to the workspace path picked at setup time:

```python
import mlflow

mlflow.set_tracking_uri("{{ tracking_uri }}")
mlflow.set_experiment("{{ experiment_path }}")
```

The path must be a workspace path such as `/Users/<email>/<name>` or
`/Shared/<team>/<name>`. MLflow creates it on first use.

**Optional: store traces in Unity Catalog.** If the user wants traces backed
by a UC Delta table (requires `mlflow[databricks]>=3.11` and a SQL warehouse),
ask for the catalog, schema, table prefix, and SQL warehouse ID, then:

```python
import mlflow
from mlflow.entities.trace_location import UnityCatalog

mlflow.set_experiment(
    experiment_name="{{ experiment_path }}",
    trace_location=UnityCatalog(
        catalog_name="<catalog>",
        schema_name="<schema>",
        table_prefix="<prefix>",
    ),
)
```

Also export `MLFLOW_TRACING_SQL_WAREHOUSE_ID=<warehouse-id>`. Skip this block
entirely if the user does not ask for UC-backed traces.
