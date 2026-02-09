# ruff: noqa: T201
from pathlib import Path

from mlflow.exceptions import MlflowException


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
    from mlflow.store.fs2db._helpers import list_experiment_ids, summary
    from mlflow.store.fs2db._registry import migrate_model_registry
    from mlflow.store.fs2db._tracking import (
        migrate_assessments,
        migrate_datasets,
        migrate_experiments,
        migrate_logged_models,
        migrate_runs,
        migrate_traces,
    )

    summary.clear()

    mlruns = source / "mlruns"
    if not mlruns.is_dir():
        # Source may be the mlruns directory itself — check if it has experiment-like subdirs
        has_experiment_dirs = any(
            d.name.isdigit() or d.name in {".trash", "models"}
            for d in source.iterdir()
            if d.is_dir()
        )
        if has_experiment_dirs:
            mlruns = source
        else:
            raise MlflowException(f"Cannot find mlruns directory in '{source}'")

    print(f"Source: {mlruns}")
    print(f"Target: {target_uri}")
    print()

    engine = create_engine(target_uri)

    print("Initializing database schema...")
    _initialize_tables(engine)
    _assert_empty_db(engine)

    with Session(engine) as session:
        try:
            # Phase 1
            print("[1/7] Migrating experiments + tags...")
            migrate_experiments(session, mlruns)
            session.flush()

            # Phase 2
            print("[2/7] Migrating runs + params + tags + metrics...")
            migrate_runs(session, mlruns)
            session.flush()

            # Phase 3
            has_datasets = any(
                (mlruns / d / "datasets").is_dir() for d in list_experiment_ids(mlruns)
            )
            if has_datasets:
                print("[3/7] Migrating datasets + inputs...")
                migrate_datasets(session, mlruns)
                session.flush()
            else:
                print("[3/7] Skipping datasets (not found)")

            # Phase 4
            has_traces = any((mlruns / d / "traces").is_dir() for d in list_experiment_ids(mlruns))
            if has_traces:
                print("[4/7] Migrating traces + tags + metadata...")
                migrate_traces(session, mlruns)
                session.flush()

                # Phase 5
                print("[5/7] Migrating assessments...")
                migrate_assessments(session, mlruns)
                session.flush()
            else:
                print("[4/7] Skipping traces (not found)")
                print("[5/7] Skipping assessments (not found)")

            # Phase 6
            has_models = any((mlruns / d / "models").is_dir() for d in list_experiment_ids(mlruns))
            if has_models:
                print("[6/7] Migrating logged models...")
                migrate_logged_models(session, mlruns)
                session.flush()
            else:
                print("[6/7] Skipping logged models (not found)")

            # Phase 7
            if (mlruns / "models").is_dir():
                print("[7/7] Migrating model registry...")
                migrate_model_registry(session, mlruns)
            else:
                print("[7/7] Skipping model registry (not found)")

            session.commit()
            print()
            print("Migration completed successfully!")

        except Exception:
            session.rollback()
            print()
            print("Migration FAILED — transaction rolled back.")
            raise

    print()
    print("=" * 50)
    print("Migration summary:")
    print("=" * 50)
    for key, count in sorted(summary.items()):
        print(f"  {key}: {count}")
    print("=" * 50)
