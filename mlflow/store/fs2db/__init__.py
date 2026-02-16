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
        for i, (exp_dir, exp_id) in enumerate(experiments, 1):
            try:
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
                session.commit()
                # Release ORM objects from the identity map to keep memory
                # proportional to one experiment, not the entire FileStore.
                session.expunge_all()
            except Exception:
                session.rollback()
                log(f"[{i}/{total}] Experiment {exp_id} FAILED — rolled back")
                raise

        # Model registry is independent of experiments
        models = list_registered_models(mlruns)
        for j, model_dir in enumerate(models, 1):
            try:
                log(f"[{j}/{len(models)}] Migrating model {model_dir.name}...")
                _migrate_one_registered_model(session, model_dir, stats)
                session.commit()
                session.expunge_all()
            except Exception:
                session.rollback()
                log(f"[{j}/{len(models)}] Model {model_dir.name} FAILED — rolled back")
                raise

    log("")
    log("Migration completed successfully!")

    print()
    print(stats.summary(str(mlruns), target_uri))
