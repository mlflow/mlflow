### Configure the Databricks workspace

You're sending traces to a Databricks workspace (`MLFLOW_TRACKING_URI={{ tracking_uri }}`).

Before instrumenting the app, verify that Databricks auth is configured.
The Databricks SDK resolves credentials from env vars, `~/.databrickscfg`
profiles, OAuth, and other sources, so don't hard-require any specific env
var: just confirm the SDK can authenticate.

```python
from databricks.sdk import WorkspaceClient

WorkspaceClient().current_user.me()
```

If that call raises, stop and ask the user to configure auth (for example via `databricks auth login`, a `~/.databrickscfg` profile, or by exporting `DATABRICKS_HOST` and `DATABRICKS_TOKEN`). Never write secrets into files in the repo.

Pin the active experiment to the workspace path picked at setup time
(tracking URI itself is wired in step 2 below, so don't repeat that here):

```python
import mlflow

mlflow.set_experiment("{{ experiment_path }}")
```

The path must be a workspace path such as `/Users/<email>/<name>` or
`/Shared/<team>/<name>`. MLflow creates it on first use.

**Optional: store traces in Unity Catalog.** If the user wants traces backed
by a UC Delta table (requires `mlflow>=3.11` and a SQL warehouse), ask for
the catalog, schema, table prefix, and SQL warehouse ID, then:

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

Skip this block entirely if the user does not ask for UC-backed traces.

#### References

If anything in this section is ambiguous, consult the authoritative Databricks
docs before guessing:

- MLflow tracing from a local IDE:
  https://docs.databricks.com/aws/en/mlflow3/genai/getting-started/tracing/tracing-ide
- Storing traces in Unity Catalog:
  https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog
- Workspace experiment paths:
  https://docs.databricks.com/aws/en/mlflow/experiments
- Databricks SDK authentication:
  https://docs.databricks.com/aws/en/dev-tools/auth/index.html
