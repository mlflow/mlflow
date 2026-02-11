# ruff: noqa: T201
import warnings
from pathlib import Path

from mlflow.exceptions import MlflowException


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


def migrate(source: Path, target_uri: str) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from mlflow.store.db.utils import _initialize_tables
    from mlflow.store.fs2db._helpers import MigrationStats, for_each_experiment
    from mlflow.store.fs2db._registry import migrate_model_registry
    from mlflow.store.fs2db._tracking import (
        _migrate_assessments_for_experiment,
        _migrate_datasets_for_experiment,
        _migrate_logged_models_for_experiment,
        _migrate_one_experiment,
        _migrate_runs_in_dir,
        _migrate_traces_for_experiment,
    )

    warnings.filterwarnings("ignore", message=".*filesystem.*deprecated.*", category=FutureWarning)

    stats = MigrationStats()
    mlruns = _resolve_mlruns(source)

    print(f"Source: {mlruns}")
    print(f"Target: {target_uri}")
    print()

    engine = create_engine(target_uri)

    print("Initializing database schema...")
    _initialize_tables(engine)
    _assert_empty_db(engine)

    with Session(engine) as session:
        for exp_dir, exp_id in for_each_experiment(mlruns):
            try:
                print(f"[experiment {exp_id}] Migrating...")
                _migrate_one_experiment(session, exp_dir, exp_id, stats)
                _migrate_runs_in_dir(session, exp_dir, int(exp_id), stats)
                session.flush()
                _migrate_datasets_for_experiment(session, exp_dir, int(exp_id), stats)
                _migrate_traces_for_experiment(session, exp_dir, int(exp_id), stats)
                session.flush()
                _migrate_assessments_for_experiment(session, exp_dir, stats)
                _migrate_logged_models_for_experiment(session, exp_dir, int(exp_id), stats)
                session.commit()
                print(f"[experiment {exp_id}] Committed")
            except Exception:
                session.rollback()
                print(f"[experiment {exp_id}] FAILED — rolled back")
                raise

        # Model registry is independent of experiments
        try:
            if (mlruns / "models").is_dir():
                print("[model registry] Migrating...")
                migrate_model_registry(session, mlruns, stats)
                session.commit()
                print("[model registry] Committed")
        except Exception:
            session.rollback()
            print("[model registry] FAILED — rolled back")
            raise

    print()
    print("Migration completed successfully!")

    print()
    print("=" * 50)
    print("Migration summary:")
    print("=" * 50)
    for key, count in stats.items():
        print(f"  {key}: {count}")
    print("=" * 50)
