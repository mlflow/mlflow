# ruff: noqa: T201
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

from mlflow.entities import RunStatus
from mlflow.store.fs2db import _resolve_mlruns
from mlflow.store.fs2db._helpers import (
    META_YAML,
    for_each_experiment,
    list_subdirs,
    read_tag_files,
    safe_read_yaml,
)
from mlflow.utils.file_utils import read_file_lines

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

RESERVED_FOLDERS = {"tags", "datasets", "traces", "models", ".trash"}


def _pass(msg: str) -> None:
    print(f"  {GREEN}PASS{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET} {msg}")


def _check_counts(conn: Connection, mlruns: Path) -> bool:
    ok = True
    tables = [
        "experiments",
        "runs",
        "trace_info",
        "assessments",
        "logged_models",
        "registered_models",
        "model_versions",
    ]
    src = _count_source(mlruns)
    for table in tables:
        expected = src[table]
        try:
            actual = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        except Exception:
            _fail(f"{table}: table not found")
            ok = False
            continue
        if actual >= expected:
            _pass(f"{table}: {actual} rows (source: {expected})")
        else:
            _fail(f"{table}: {actual} < {expected} (source)")
            ok = False
    return ok


def _count_source(mlruns: Path) -> dict[str, int]:
    n = {
        "experiments": 0,
        "runs": 0,
        "trace_info": 0,
        "assessments": 0,
        "logged_models": 0,
        "registered_models": 0,
        "model_versions": 0,
    }
    for exp_dir, _exp_id in for_each_experiment(mlruns):
        if not (exp_dir / META_YAML).is_file():
            continue
        n["experiments"] += 1
        for name in list_subdirs(exp_dir):
            if name not in RESERVED_FOLDERS and (exp_dir / name / META_YAML).is_file():
                n["runs"] += 1
        traces_dir = exp_dir / "traces"
        if traces_dir.is_dir():
            for td in list_subdirs(traces_dir):
                if (traces_dir / td / "trace_info.yaml").is_file():
                    n["trace_info"] += 1
                    adir = traces_dir / td / "assessments"
                    if adir.is_dir():
                        n["assessments"] += sum(1 for f in adir.iterdir() if f.suffix == ".yaml")
        models_dir = exp_dir / "models"
        if models_dir.is_dir():
            for md in list_subdirs(models_dir):
                if (models_dir / md / META_YAML).is_file():
                    n["logged_models"] += 1
    registry_dir = mlruns / "models"
    if registry_dir.is_dir():
        for mn in list_subdirs(registry_dir):
            if (registry_dir / mn / META_YAML).is_file():
                n["registered_models"] += 1
                for vd in list_subdirs(registry_dir / mn):
                    if vd.startswith("version-"):
                        n["model_versions"] += 1
    return n


def _check_experiments(conn: Connection, mlruns: Path) -> bool:
    ok = True
    for exp_dir, exp_id in for_each_experiment(mlruns):
        meta = safe_read_yaml(exp_dir, META_YAML)
        if meta is None:
            continue
        row = conn.execute(
            text("SELECT name, lifecycle_stage FROM experiments WHERE experiment_id = :id"),
            {"id": int(exp_id)},
        ).first()
        if row is None:
            _fail(f"experiment {exp_id}: missing from DB")
            ok = False
            continue
        if row[0] != meta.get("name"):
            _fail(f"experiment {exp_id}: name {row[0]!r} != {meta.get('name')!r}")
            ok = False
        elif row[1] != meta.get("lifecycle_stage", "active"):
            expected = meta.get("lifecycle_stage")
            _fail(f"experiment {exp_id}: lifecycle_stage {row[1]!r} != {expected!r}")
            ok = False
        else:
            _pass(f"experiment {exp_id} ({row[0]})")
        break  # spot-check first only
    return ok


def _check_runs(conn: Connection, mlruns: Path) -> bool:
    ok = True
    for exp_dir, _exp_id in for_each_experiment(mlruns):
        for name in list_subdirs(exp_dir):
            if name in RESERVED_FOLDERS:
                continue
            meta = safe_read_yaml(exp_dir / name, META_YAML)
            if meta is None:
                continue
            run_uuid = meta.get("run_uuid") or meta.get("run_id")
            if not run_uuid:
                continue
            row = conn.execute(
                text("SELECT status, lifecycle_stage, artifact_uri FROM runs WHERE run_uuid = :id"),
                {"id": run_uuid},
            ).first()
            if row is None:
                _fail(f"run {run_uuid}: missing from DB")
                ok = False
                continue
            status_raw = meta.get("status", RunStatus.RUNNING)
            expected_status = (
                RunStatus.to_string(status_raw) if isinstance(status_raw, int) else str(status_raw)
            )
            if row[0] != expected_status:
                _fail(f"run {run_uuid}: status {row[0]!r} != {expected_status!r}")
                ok = False
            elif row[1] != meta.get("lifecycle_stage", "active"):
                _fail(f"run {run_uuid}: lifecycle_stage {row[1]!r}")
                ok = False
            else:
                _pass(f"run {run_uuid} (status={row[0]}, artifacts={'yes' if row[2] else 'no'})")
            return ok  # spot-check first run
    return ok


def _check_metrics(conn: Connection, mlruns: Path) -> bool:
    ok = True
    for exp_dir, _exp_id in for_each_experiment(mlruns):
        for name in list_subdirs(exp_dir):
            if name in RESERVED_FOLDERS:
                continue
            metrics_dir = exp_dir / name / "metrics"
            if not metrics_dir.is_dir():
                continue
            for metric_file in metrics_dir.iterdir():
                if not metric_file.is_file():
                    continue
                key = metric_file.name
                lines = read_file_lines(str(metrics_dir), key)
                file_count = len(lines)
                run_uuid = name
                db_count = conn.execute(
                    text("SELECT COUNT(*) FROM metrics WHERE run_uuid = :id AND key = :key"),
                    {"id": run_uuid, "key": key},
                ).scalar()
                if db_count != file_count:
                    _fail(f"metrics {run_uuid}/{key}: DB has {db_count}, file has {file_count}")
                    ok = False
                else:
                    _pass(f"metrics {run_uuid}/{key}: {db_count} values")
                return ok  # spot-check first metric
    return ok


def _check_params(conn: Connection, mlruns: Path) -> bool:
    ok = True
    for exp_dir, _exp_id in for_each_experiment(mlruns):
        for name in list_subdirs(exp_dir):
            if name in RESERVED_FOLDERS:
                continue
            params = read_tag_files(exp_dir / name / "params")
            if not params:
                continue
            run_uuid = name
            for key, file_val in params.items():
                row = conn.execute(
                    text("SELECT value FROM params WHERE run_uuid = :id AND key = :key"),
                    {"id": run_uuid, "key": key},
                ).first()
                if row is None:
                    _fail(f"param {run_uuid}/{key}: missing from DB")
                    ok = False
                elif row[0] != file_val:
                    _fail(f"param {run_uuid}/{key}: {row[0]!r} != {file_val!r}")
                    ok = False
                else:
                    _pass(f"param {run_uuid}/{key} = {row[0]!r}")
                return ok  # spot-check first param
    return ok


def _check_registered_models(conn: Connection, mlruns: Path) -> bool:
    ok = True
    registry_dir = mlruns / "models"
    if not registry_dir.is_dir():
        return ok
    for model_name in list_subdirs(registry_dir):
        meta = safe_read_yaml(registry_dir / model_name, META_YAML)
        if meta is None:
            continue
        name = meta.get("name", model_name)
        row = conn.execute(
            text("SELECT description FROM registered_models WHERE name = :name"),
            {"name": name},
        ).first()
        if row is None:
            _fail(f"registered_model {name}: missing from DB")
            ok = False
        else:
            _pass(f"registered_model {name} (description={row[0]!r})")
        return ok  # spot-check first model
    return ok


def verify_migration(source: Path, target_uri: str) -> None:
    mlruns = _resolve_mlruns(source)
    engine = create_engine(target_uri)

    print()
    print("Verification:")
    ok = True
    with engine.connect() as conn:
        print("  -- Row counts --")
        ok &= _check_counts(conn, mlruns)
        print("  -- Spot checks --")
        ok &= _check_experiments(conn, mlruns)
        ok &= _check_runs(conn, mlruns)
        ok &= _check_params(conn, mlruns)
        ok &= _check_metrics(conn, mlruns)
        ok &= _check_registered_models(conn, mlruns)

    print()
    if ok:
        print(f"{GREEN}Verification passed{RESET}")
    else:
        print(f"{RED}Verification failed{RESET}")
        raise SystemExit(1)
