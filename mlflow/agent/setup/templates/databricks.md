### Configure the Databricks workspace

You're sending traces to a Databricks workspace (`MLFLOW_TRACKING_URI={{ tracking_uri }}`).
When installing MLflow in step 1, prefer `mlflow[databricks]` over plain
`mlflow` so the Databricks SDK extras are pulled in.

Before instrumenting the app, verify that Databricks auth is configured.
The Databricks SDK resolves credentials from env vars, `~/.databrickscfg`
profiles, OAuth, and other sources, so don't hard-require any specific env
var: just confirm the SDK can authenticate.

```python
from databricks.sdk import WorkspaceClient

WorkspaceClient().current_user.me()
```

If that call raises, stop and ask the user to configure auth (for example via `databricks auth login`, a `~/.databrickscfg` profile, or by exporting `DATABRICKS_HOST` and `DATABRICKS_TOKEN`). Never write secrets into files in the repo.

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
