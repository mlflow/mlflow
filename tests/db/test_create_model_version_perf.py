"""
Verification script for create_model_version performance and concurrency fix.

Tests against a real PostgreSQL database to prove:
1. OLD approach (Python max over lazy-loaded rows) is slow with many versions
2. NEW approach (SQL MAX query) is fast regardless of row count
3. Concurrent create_model_version works correctly with session.rollback() fix

Usage:
    uv run python tests/db/test_create_model_version_perf.py
"""

import atexit
import concurrent.futures
import socket
import statistics
import subprocess
import sys
import threading
import time
import uuid

import sqlalchemy
from sqlalchemy import event, insert

from mlflow.store.model_registry.dbmodels.models import SqlModelVersion, SqlRegisteredModel
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

PG_IMAGE = "postgres@sha256:c1f0abd909b477d6088c72e4cd6eb01ea525344caca1b58689ae884204369502"
CONTAINER_NAME = "mlflow-perf-test-pg"
PG_DB = "mlflowdb"
PG_USER = "mlflowuser"
PG_PASSWORD = "mlflowpassword"

NUM_VERSIONS = 1_000_000
BATCH_SIZE = 5_000
BENCHMARK_ITERATIONS = 5
CONCURRENT_THREADS = 8
VERSIONS_PER_THREAD = 10
MODEL_NAME = "perf-test-model"


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def cleanup_container():
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
    )


def start_postgres(port):
    cleanup_container()
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"{port}:5432",
            "-e",
            f"POSTGRES_DB={PG_DB}",
            "-e",
            f"POSTGRES_USER={PG_USER}",
            "-e",
            f"POSTGRES_PASSWORD={PG_PASSWORD}",
            PG_IMAGE,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"Failed to start PostgreSQL container: {result.stderr}")
    atexit.register(cleanup_container)

    # Wait for PostgreSQL to accept connections via SQLAlchemy
    db_uri = f"postgresql://{PG_USER}:{PG_PASSWORD}@localhost:{port}/{PG_DB}"
    for i in range(60):
        try:
            eng = sqlalchemy.create_engine(db_uri)
            with eng.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            eng.dispose()
            return
        except Exception:
            time.sleep(1)
    sys.exit("PostgreSQL not ready after 60s")


def bulk_insert_versions(store, engine, model_name, num_versions):
    now = get_current_time_millis()
    with engine.begin() as conn:
        for batch_start in range(1, num_versions + 1, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, num_versions + 1)
            rows = [
                {
                    "workspace": DEFAULT_WORKSPACE_NAME,
                    "name": model_name,
                    "version": v,
                    "creation_time": now,
                    "last_updated_time": now,
                    "source": "s3://dummy",
                    "current_stage": "None",
                    "status": "READY",
                    "storage_location": "s3://dummy",
                }
                for v in range(batch_start, batch_end)
            ]
            conn.execute(insert(SqlModelVersion), rows)


def benchmark_old_approach(engine, model_name):
    """Old: loads all rows via ORM relationship, computes max in Python."""
    Session = sqlalchemy.orm.sessionmaker(bind=engine)
    session = Session()
    try:
        rm = (
            session.query(SqlRegisteredModel)
            .filter(
                SqlRegisteredModel.name == model_name,
                SqlRegisteredModel.workspace == DEFAULT_WORKSPACE_NAME,
            )
            .one()
        )
        if rm.model_versions:
            return max(mv.version for mv in rm.model_versions) + 1
        return 1
    finally:
        session.close()


def benchmark_new_approach(engine, model_name):
    """New: single SELECT MAX(version) query using PK index."""
    Session = sqlalchemy.orm.sessionmaker(bind=engine)
    session = Session()
    try:
        max_version = (
            session.query(sqlalchemy.func.max(SqlModelVersion.version))
            .filter(
                SqlModelVersion.name == model_name,
                SqlModelVersion.workspace == DEFAULT_WORKSPACE_NAME,
            )
            .scalar()
        )
        return (max_version or 0) + 1
    finally:
        session.close()


def count_queries(engine, func, *args):
    count = 0

    def _before_execute(conn, cursor, statement, parameters, context, executemany):
        nonlocal count
        count += 1

    event.listen(engine, "before_cursor_execute", _before_execute)
    try:
        result = func(engine, *args)
        return result, count
    finally:
        event.remove(engine, "before_cursor_execute", _before_execute)


def run_benchmark(engine, model_name, approach_name, func, iterations):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(engine, model_name)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Count queries on one extra run
    result, query_count = count_queries(engine, func, model_name)

    print(f"\n  {approach_name}:")
    print(f"    Iterations: {iterations}")
    print(f"    Times: [{', '.join(f'{t:.3f}s' for t in times)}]")
    print(f"    Mean: {statistics.mean(times):.3f}s | Median: {statistics.median(times):.3f}s")
    print(f"    SQL queries: {query_count}")
    print(f"    Computed next version: {result}")
    return statistics.mean(times)


def main():
    # Check docker
    r = subprocess.run(["docker", "info"], capture_output=True)
    if r.returncode != 0:
        sys.exit("Docker is not available. Please start Docker.")

    port = find_free_port()

    print("=" * 60)
    print("  create_model_version Performance & Concurrency Test")
    print("  PostgreSQL verification")
    print("=" * 60)

    # Phase 0: Start PostgreSQL
    print("\n=== Phase 0: Start PostgreSQL ===")
    start_postgres(port)
    db_uri = f"postgresql://{PG_USER}:{PG_PASSWORD}@localhost:{port}/{PG_DB}"
    print(f"  PostgreSQL running on port {port}")

    # Create store (initializes tables via Alembic)
    store = SqlAlchemyStore(db_uri)
    engine = sqlalchemy.create_engine(db_uri)

    # Phase 1: Setup
    print("\n=== Phase 1: Bulk Data Population ===")
    store.create_registered_model(MODEL_NAME)
    print(f"  Inserting {NUM_VERSIONS:,} model versions...")
    t0 = time.perf_counter()
    bulk_insert_versions(store, engine, MODEL_NAME, NUM_VERSIONS)
    t1 = time.perf_counter()
    print(f"  Done in {t1 - t0:.1f}s")

    # Verify count
    with engine.connect() as conn:
        count = conn.execute(
            sqlalchemy.text(
                f"SELECT COUNT(*) FROM model_versions WHERE name = '{MODEL_NAME}'"
            )
        ).scalar()
    print(f"  Verified: {count:,} rows in model_versions")

    # Phase 2: Benchmark
    print("\n=== Phase 2: Performance Benchmark ===")
    old_mean = run_benchmark(
        engine, MODEL_NAME, "OLD (Python max over lazy-loaded rows)",
        benchmark_old_approach, BENCHMARK_ITERATIONS,
    )
    new_mean = run_benchmark(
        engine, MODEL_NAME, "NEW (SQL MAX query)",
        benchmark_new_approach, BENCHMARK_ITERATIONS,
    )

    speedup = old_mean / new_mean if new_mean > 0 else float("inf")
    print(f"\n  Speedup: {speedup:,.0f}x")

    # Phase 3: Concurrency Test
    print("\n=== Phase 3: Concurrency Test ===")
    total_expected = CONCURRENT_THREADS * VERSIONS_PER_THREAD
    print(f"  Threads: {CONCURRENT_THREADS} | Versions/thread: {VERSIONS_PER_THREAD} | Total: {total_expected}")

    results = []
    errors = []
    lock = threading.Lock()

    def create_versions():
        thread_versions = []
        for _ in range(VERSIONS_PER_THREAD):
            try:
                mv = store.create_model_version(MODEL_NAME, "s3://dummy", uuid.uuid4().hex)
                with lock:
                    thread_versions.append(mv.version)
            except Exception as e:
                with lock:
                    errors.append(e)
        return thread_versions

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
        futures = [executor.submit(create_versions) for _ in range(CONCURRENT_THREADS)]
        for f in concurrent.futures.as_completed(futures):
            results.extend(f.result())
    t1 = time.perf_counter()

    has_errors = len(errors) > 0
    has_duplicates = len(results) != len(set(results))
    all_created = len(results) == total_expected

    print(f"  Completed in {t1 - t0:.1f}s")
    print(f"  Errors: {len(errors)}" + (f" — {errors[:3]}" if errors else ""))
    print(f"  Versions created: {len(results)}/{total_expected}")
    print(f"  Unique versions: {len(set(results))}")
    if results:
        print(f"  Version range: {min(results)} - {max(results)}")
    print(f"  Duplicates: {'YES — FAIL' if has_duplicates else 'NONE'}")

    # Verify DB count
    with engine.connect() as conn:
        final_count = conn.execute(
            sqlalchemy.text(
                f"SELECT COUNT(*) FROM model_versions WHERE name = '{MODEL_NAME}'"
            )
        ).scalar()
    expected_total = NUM_VERSIONS + total_expected
    print(f"  DB total row count: {final_count:,} (expected: {expected_total:,})")

    # Summary
    print("\n" + "=" * 60)
    perf_pass = speedup > 10
    concurrency_pass = not has_errors and not has_duplicates and all_created
    db_pass = final_count == expected_total

    print(f"  Performance (>{10}x speedup):     {'PASS' if perf_pass else 'FAIL'} ({speedup:,.0f}x)")
    print(f"  Concurrency (no dupes/errors): {'PASS' if concurrency_pass else 'FAIL'}")
    print(f"  DB consistency:                {'PASS' if db_pass else 'FAIL'}")
    print("=" * 60)

    if perf_pass and concurrency_pass and db_pass:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED")
        sys.exit(1)

    engine.dispose()


if __name__ == "__main__":
    main()
