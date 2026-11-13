import io
import tarfile
from pathlib import Path

import pytest

from mlflow.server import ARTIFACTS_DESTINATION_ENV_VAR, SERVE_ARTIFACTS_ENV_VAR, handlers
from mlflow.store.tracking.dbmodels.models import SqlSkillVersion
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

SKILL_FILES = {
    "SKILL.md": b"---\nname: reviewer\ndescription: Reviews code\n---\n# Reviewer\n",
    "scripts/run.py": b"print('hi')\n",
}


def skill_archive(files=None, *, extra_members=()) -> io.BytesIO:
    """A gzip tar of ``files`` (path -> bytes), plus raw ``TarInfo`` members for bad cases."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name, data in (SKILL_FILES if files is None else files).items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            tar.addfile(entry, io.BytesIO(data))
        for member in extra_members:
            tar.addfile(member)
    out.seek(0)
    return out


@pytest.fixture
def store(tmp_path: Path, db_uri: str, monkeypatch) -> SqlAlchemyStore:
    sql_store = SqlAlchemyStore(db_uri, (tmp_path / "run-artifacts").as_uri())
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda *args, **kwargs: sql_store)
    return sql_store


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch) -> Path:
    """A server that serves artifacts from a local directory."""
    root = tmp_path / "artifacts-destination"
    root.mkdir()
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "true")
    monkeypatch.setenv(ARTIFACTS_DESTINATION_ENV_VAR, str(root))
    # The handlers cache the repository for the process; each test gets its own root.
    monkeypatch.setattr(handlers, "_artifact_repo", None)
    return root


@pytest.fixture
def no_artifact_serving(monkeypatch) -> None:
    """A server started without ``--serve-artifacts``."""
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "false")
    monkeypatch.delenv(ARTIFACTS_DESTINATION_ENV_VAR, raising=False)
    monkeypatch.setattr(handlers, "_artifact_repo", None)


def version_rows(store: SqlAlchemyStore) -> int:
    with store.ManagedSessionMaker() as session:
        return session.query(SqlSkillVersion).count()


def stored_files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
