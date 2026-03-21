# MLflow Tracking database migrations

This directory contains configuration scripts and database migration logic for MLflow tracking
databases, using the Alembic migration library (https://alembic.sqlalchemy.org). To run database
migrations, use the `mlflow db upgrade` CLI command. To add and modify database migration logic,
see the contributor guide at https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md.

If you encounter failures while executing migrations, please file a GitHub issue at
https://github.com/mlflow/mlflow/issues.

## Migration descriptions

### 89d4b8295536_create_latest_metrics_table

This migration creates a `latest_metrics` table and populates it with the latest metric entry for
each unique `(run_id, metric_key)` tuple. Latest metric entries are computed based on `step`,
`timestamp`, and `value`.

This migration may take a long time for databases containing a large number of metric entries. You
can determine the total number of metric entries using the following query:

```sql
SELECT count(*) FROM metrics GROUP BY metrics.key, run_uuid;
```

Additionally, query join latency during the migration increases with the number of unique
`(run_id, metric_key)` tuples. You can determine the total number of unique tuples using
the following query:

```sql
SELECT count(*) FROM (
   SELECT metrics.key, run_uuid FROM metrics GROUP BY run_uuid, metrics.key
) unique_metrics;
```

For reference, migrating a Tracking database with the following attributes takes roughly
**three seconds** on MySQL 5.7:

- `3702` unique metrics
- `466860` total metric entries
- `186` runs
- An average of `125` entries per unique metric

#### Recovering from a failed migration

If the **create_latest_metrics_table** migration fails, simply delete the `latest_metrics`
table from your Tracking database as follows:

```sql
DROP TABLE latest_metrics;
```

Alembic does not stamp the database with an updated version unless the corresponding migration
completes successfully. Therefore, when this migration fails, the database remains on the
previous version, and deleting the `latest_metrics` table is sufficient to restore the database
to its prior state.

If the migration fails to complete due to excessive latency, please try executing the
`mlflow db upgrade` command on the same host machine where the database is running. This will
reduce the overhead of the migration's queries and batch insert operation.
