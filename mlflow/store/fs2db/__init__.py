# ruff: noqa: T201
import warnings
from functools import partial
from pathlib import Path

from mlflow.exceptions import MlflowException


def _log(progress: bool, msg: str) -> None:
    if progress:
        print(msg)


def _resolve_mlruns(source: Path) -> Path:
    mlruns = source / "mlruns"
    if mlruns.is_dir():
        return mlruns

    has_experiment_dirs = any(
        d.name.isdigit() or d.name in {".trash", "models"} for d in source.iterdir() if d.is_dir()
    )
    if has_experiment_dirs:
        return source

    raise MlflowException(f"Cannot find mlruns directory in '{source}'")


def _assert_empty_db(engine) -> None:
    from sqlalchemy import text

    with engine.connect() as conn:
        for table in ("experiments", "runs", "registered_models"):
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            except Exception:
                continue
            if count > 0:
                raise MlflowException(
                    f"Target database is not empty: table '{table}' has {count} rows. "
                    "Migration requires an empty database."
                )


_ROW_COUNT_QUERIES: dict[str, str] = {
    "experiments": "SELECT COUNT(*) FROM experiments",
    "experiment_tags": "SELECT COUNT(*) FROM experiment_tags",
    "runs": "SELECT COUNT(*) FROM runs",
    "params": "SELECT COUNT(*) FROM params",
    "tags": "SELECT COUNT(*) FROM tags",
    "metrics": "SELECT COUNT(*) FROM metrics",
    "latest_metrics": "SELECT COUNT(*) FROM latest_metrics",
    "datasets": "SELECT COUNT(*) FROM datasets",
    "inputs": "SELECT COUNT(*) FROM inputs WHERE source_type = 'DATASET'",
    "input_tags": "SELECT COUNT(*) FROM input_tags",
    "outputs": "SELECT COUNT(*) FROM inputs WHERE source_type = 'RUN_OUTPUT'",
    "traces": "SELECT COUNT(*) FROM trace_info",
    "trace_tags": "SELECT COUNT(*) FROM trace_tags",
    "trace_metadata": "SELECT COUNT(*) FROM trace_request_metadata",
    "assessments": "SELECT COUNT(*) FROM assessments",
    "logged_models": "SELECT COUNT(*) FROM logged_models",
    "logged_model_params": "SELECT COUNT(*) FROM logged_model_params",
    "logged_model_tags": "SELECT COUNT(*) FROM logged_model_tags",
    "logged_model_metrics": "SELECT COUNT(*) FROM logged_model_metrics",
    "registered_models": "SELECT COUNT(*) FROM registered_models",
    "registered_model_tags": "SELECT COUNT(*) FROM registered_model_tags",
    "registered_model_aliases": "SELECT COUNT(*) FROM registered_model_aliases",
    "model_versions": "SELECT COUNT(*) FROM model_versions",
    "model_version_tags": "SELECT COUNT(*) FROM model_version_tags",
}


def _query_row_counts(engine) -> dict[str, int]:
    from sqlalchemy import text

    counts = {}
    with engine.connect() as conn:
        for key, query in _ROW_COUNT_QUERIES.items():
            try:
                counts[key] = conn.execute(text(query)).scalar()
            except Exception:
                pass
    return counts


def migrate(source: Path, target_uri: str, *, progress: bool = True) -> None:
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import Session

    from mlflow.store.db.utils import _initialize_tables
    from mlflow.store.fs2db._registry import _migrate_one_registered_model, list_registered_models
    from mlflow.store.fs2db._tracking import (
        _migrate_assessments_for_experiment,
        _migrate_datasets_for_experiment,
        _migrate_logged_models_for_experiment,
        _migrate_one_experiment,
        _migrate_outputs_for_experiment,
        _migrate_runs_in_dir,
        _migrate_traces_for_experiment,
    )
    from mlflow.store.fs2db._utils import MigrationStats, for_each_experiment

    log = partial(_log, progress)

    warnings.filterwarnings("ignore", message=".*filesystem.*deprecated.*", category=FutureWarning)

    stats = MigrationStats()
    mlruns = _resolve_mlruns(source)

    log(f"Source: {mlruns}")
    log(f"Target: {target_uri}")
    log("")

    engine = create_engine(target_uri)

    # Optimize SQLite for bulk import: WAL mode reduces lock contention,
    # synchronous=OFF skips fsync (safe here since we can re-run on failure).
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=OFF")
        cursor.close()

    log("Initializing database schema...")
    _initialize_tables(engine)
    _assert_empty_db(engine)

    experiments = list(for_each_experiment(mlruns))
    total = len(experiments)

    with Session(engine) as session:
        try:
            for i, (exp_dir, exp_id) in enumerate(experiments, 1):
                log(f"[{i}/{total}] Migrating experiment {exp_id}...")
                _migrate_one_experiment(session, exp_dir, exp_id, stats)
                _migrate_runs_in_dir(session, exp_dir, int(exp_id), stats)
                session.flush()
                _migrate_datasets_for_experiment(session, exp_dir, int(exp_id), stats)
                _migrate_outputs_for_experiment(session, exp_dir, stats)
                _migrate_traces_for_experiment(session, exp_dir, int(exp_id), stats)
                session.flush()
                _migrate_assessments_for_experiment(session, exp_dir, stats)
                _migrate_logged_models_for_experiment(session, exp_dir, int(exp_id), stats)
                session.flush()
                session.expunge_all()

            # Model registry is independent of experiments
            models = list_registered_models(mlruns)
            for j, model_dir in enumerate(models, 1):
                log(f"[{j}/{len(models)}] Migrating model {model_dir.name}...")
                _migrate_one_registered_model(session, model_dir, stats)
                session.flush()
                session.expunge_all()

            session.commit()
        except Exception:
            session.rollback()
            raise

    log("")
    log("Migration completed successfully!")

    db_counts = _query_row_counts(engine)
    print()
    print(stats.summary(str(mlruns), target_uri, db_counts))
