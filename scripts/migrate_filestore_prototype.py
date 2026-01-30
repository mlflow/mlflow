# ruff: noqa: T201
"""
Minimal prototype: FileStore to SQLite migration.
Only migrates experiments and runs (no params, tags, metrics).

Usage:
    python scripts/migrate_filestore_prototype.py
"""

import os
import tempfile

import sqlalchemy
from sqlalchemy.orm import sessionmaker

import mlflow
from mlflow.entities import ViewType
from mlflow.store.db.utils import _initialize_tables
from mlflow.store.tracking.dbmodels.models import SqlExperiment, SqlRun
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


def migrate(source_path: str, target_uri: str):
    """Migrate FileStore data to SQLite."""
    file_store = FileStore(source_path)

    engine = sqlalchemy.create_engine(target_uri)
    _initialize_tables(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        for exp in file_store.search_experiments(view_type=ViewType.ALL):
            print(f"Migrating experiment: {exp.name} (ID: {exp.experiment_id})")

            sql_exp = SqlExperiment(
                experiment_id=int(exp.experiment_id),
                name=exp.name,
                artifact_location=exp.artifact_location,
                lifecycle_stage=exp.lifecycle_stage,
                creation_time=exp.creation_time,
                last_update_time=exp.last_update_time,
            )
            session.add(sql_exp)

            for run in file_store.search_runs([exp.experiment_id], "", ViewType.ALL):
                print(f"  Migrating run: {run.info.run_id}")

                sql_run = SqlRun(
                    run_uuid=run.info.run_id,
                    name=run.info.run_name,
                    experiment_id=int(exp.experiment_id),
                    user_id=run.info.user_id,
                    status=run.info.status,
                    start_time=run.info.start_time,
                    end_time=run.info.end_time,
                    lifecycle_stage=run.info.lifecycle_stage,
                    artifact_uri=run.info.artifact_uri,
                )
                session.add(sql_run)

        session.commit()
    print("Migration complete!")


def verify(source_path: str, target_uri: str):
    """Verify migration by comparing source and target."""
    file_store = FileStore(source_path)
    sql_store = SqlAlchemyStore(target_uri, "./mlartifacts")

    print("\n=== Verification ===")

    src_exps = {e.experiment_id: e for e in file_store.search_experiments(view_type=ViewType.ALL)}
    dst_exps = {e.experiment_id: e for e in sql_store.search_experiments(view_type=ViewType.ALL)}

    assert src_exps.keys() == dst_exps.keys(), "Experiment IDs mismatch"
    print(f"Experiments: {len(src_exps)} OK")

    for exp_id in src_exps:
        src_runs = {r.info.run_id: r for r in file_store.search_runs([exp_id], "", ViewType.ALL)}
        dst_runs = {r.info.run_id: r for r in sql_store.search_runs([exp_id], "", ViewType.ALL)}

        assert src_runs.keys() == dst_runs.keys(), f"Run IDs mismatch for exp {exp_id}"
        print(f"  Experiment {exp_id}: {len(src_runs)} runs OK")

    print("\nVerification passed!")


def create_test_data(source_path: str):
    """Create test FileStore data."""
    mlflow.set_tracking_uri(source_path)
    mlflow.set_experiment("test_experiment")

    with mlflow.start_run(run_name="run1"):
        pass

    with mlflow.start_run(run_name="run2"):
        pass

    print(f"Test data created at: {source_path}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "mlruns")
        target_uri = f"sqlite:///{os.path.join(tmpdir, 'mlflow.db')}"

        print(f"Source: {source_path}")
        print(f"Target: {target_uri}\n")

        create_test_data(source_path)
        migrate(source_path, target_uri)
        verify(source_path, target_uri)
