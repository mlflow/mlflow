# MLflow Tracking database migrations

This directory contains configuration scripts and database migration logic for MLflow tracking
databases, using the Alembic migration library (https://alembic.sqlalchemy.org). To run database
migrations, use the ``mlflow db upgrade`` CLI command. To add and modify database migration logic,
see the contributor guide at https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.rst.

## Migration descriptions

### 89d4b8295536_create_latest_metrics_table
This migration creates a ``latest_metrics`` table and populates it with the latest metric entry for 
each ``(run_id, metric_key)`` tuple. Latest metric entries are computed based on step, timestamp, 
and value. 

This migration may take a long time for databases containing a large number of metric entries.

If you encounter failures while executing this migration, please file a GitHub issue at
https://github.com/mlflow/mlflow/issues.


