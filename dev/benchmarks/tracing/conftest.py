from collections.abc import Iterator
from pathlib import Path

import pytest
from _data import SEED_SPANS_PER_TRACE, SEED_TRACES, seed_traces

import mlflow
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


@pytest.fixture(scope="session")
def bench_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("bench")


@pytest.fixture(scope="session")
def store(bench_dir: Path) -> SqlAlchemyStore:
    db_uri = f"sqlite:///{bench_dir / 'mlflow.db'}"
    (bench_dir / "artifacts").mkdir()
    artifact_root = (bench_dir / "artifacts").as_uri()
    return SqlAlchemyStore(db_uri, artifact_root)


@pytest.fixture(scope="session")
def experiment_id(store: SqlAlchemyStore) -> str:
    return str(store.create_experiment("bench"))


@pytest.fixture(scope="session")
def seeded(store: SqlAlchemyStore, experiment_id: str) -> list[str]:
    return seed_traces(store, experiment_id, SEED_TRACES, SEED_SPANS_PER_TRACE)


@pytest.fixture(scope="session")
def e2e_setup(bench_dir: Path) -> Iterator[None]:
    mlflow.set_tracking_uri(f"sqlite:///{bench_dir / 'e2e.db'}")
    mlflow.set_experiment("bench_e2e")
    yield
    mlflow.flush_trace_async_logging(terminate=True)
