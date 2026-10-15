import json

import sqlalchemy as sa
from alembic import command

from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


def test_issue_contract_migration_preserves_existing_issues(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'issue-migration.db'}"
    engine = sa.create_engine(db_url)
    InitialBase.metadata.create_all(engine)
    config = _get_alembic_config(db_url)
    command.upgrade(config, "dc11669786a5")
    tables = sa.MetaData()
    tables.reflect(engine)
    with engine.begin() as conn:
        conn.execute(
            tables
            .tables["experiments"]
            .insert()
            .values(
                experiment_id=1,
                name="legacy",
                artifact_location="",
                lifecycle_stage="active",
                workspace="default",
            )
        )
        conn.execute(
            tables
            .tables["runs"]
            .insert()
            .values(run_uuid="old-run", experiment_id=1, lifecycle_stage="active")
        )
        conn.execute(
            tables
            .tables["issues"]
            .insert()
            .values(
                issue_id="iss-old",
                experiment_id=1,
                name="Legacy issue",
                description="Description",
                status="pending",
                severity=None,
                root_causes=json.dumps(["Old cause"]),
                categories=json.dumps(["cost", "cost", "c" * 300]),
                created_timestamp=100,
                last_updated_timestamp=200,
                source_run_id="old-run",
                created_by="author",
            )
        )
    command.upgrade(config, "8b6ad64cb8f5")
    store = SqlAlchemyStore(db_url, str(tmp_path / "artifacts"))
    issue = store.get_issue("iss-old")
    assert issue.last_seen_timestamp == 100
    assert issue.last_updated_timestamp == 200
    assert issue.last_updated_by is None
    assert issue.root_causes == ["Old cause"]
    assert issue.categories == ["cost", "cost", "c" * 300]
    assert issue.source_run_id == "old-run"
    assert issue.severity is None
    assert [result.issue_id for result in store.search_issues(categories=["cost"])] == ["iss-old"]
    assert store.update_issue("iss-old", categories=["latency"]).categories == ["latency"]
    assert store.search_issues(categories=["cost"]) == []
