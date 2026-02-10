# ruff: noqa: T201
from pathlib import Path

from mlflow.store.fs2db._helpers import (
    META_YAML,
    for_each_experiment,
    list_files,
    list_subdirs,
)

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

RESERVED_FOLDERS = {"tags", "datasets", "traces", "models", ".trash"}


def _count_source(mlruns: Path) -> dict[str, int]:
    counts: dict[str, int] = {}

    n_experiments = 0
    n_runs = 0
    n_traces = 0
    n_assessments = 0
    n_logged_models = 0
    n_registered_models = 0
    n_model_versions = 0

    for exp_dir, _exp_id in for_each_experiment(mlruns):
        if not (exp_dir / META_YAML).is_file():
            continue
        n_experiments += 1

        # Runs
        for name in list_subdirs(exp_dir):
            if name in RESERVED_FOLDERS:
                continue
            if (exp_dir / name / META_YAML).is_file():
                n_runs += 1

        # Traces
        traces_dir = exp_dir / "traces"
        if traces_dir.is_dir():
            for td in list_subdirs(traces_dir):
                if (traces_dir / td / "trace_info.yaml").is_file():
                    n_traces += 1

                    # Assessments
                    assessments_dir = traces_dir / td / "assessments"
                    if assessments_dir.is_dir():
                        n_assessments += sum(
                            1 for f in list_files(assessments_dir) if f.endswith(".yaml")
                        )

        # Logged models
        models_dir = exp_dir / "models"
        if models_dir.is_dir():
            for md in list_subdirs(models_dir):
                if (models_dir / md / META_YAML).is_file():
                    n_logged_models += 1

    # Model registry
    registry_dir = mlruns / "models"
    if registry_dir.is_dir():
        for model_name in list_subdirs(registry_dir):
            if (registry_dir / model_name / META_YAML).is_file():
                n_registered_models += 1
                for vd in list_subdirs(registry_dir / model_name):
                    if vd.startswith("version-"):
                        n_model_versions += 1

    counts["experiments"] = n_experiments
    counts["runs"] = n_runs
    counts["trace_info"] = n_traces
    counts["assessments"] = n_assessments
    counts["logged_models"] = n_logged_models
    counts["registered_models"] = n_registered_models
    counts["model_versions"] = n_model_versions
    return counts


def _count_db(target_uri: str, tables: list[str]) -> dict[str, int | None]:
    from sqlalchemy import create_engine, text

    engine = create_engine(target_uri)
    counts: dict[str, int | None] = {}
    with engine.connect() as conn:
        for table in tables:
            try:
                counts[table] = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            except Exception:
                counts[table] = None
    return counts


def verify_migration(source: Path, target_uri: str) -> None:
    from mlflow.store.fs2db import _resolve_mlruns

    mlruns = _resolve_mlruns(source)
    src = _count_source(mlruns)
    db = _count_db(target_uri, list(src.keys()))

    print()
    print("Verification:")
    ok = True
    for table, expected in src.items():
        actual = db.get(table)
        if actual is None:
            print(f"  {RED}FAIL{RESET} {table}: table not found")
            ok = False
        elif actual >= expected:
            print(f"  {GREEN}PASS{RESET} {table}: {actual} (source: {expected})")
        else:
            print(f"  {RED}FAIL{RESET} {table}: {actual} < {expected} (source)")
            ok = False

    print()
    if ok:
        print(f"{GREEN}Verification passed{RESET}")
    else:
        print(f"{RED}Verification failed{RESET}")
        raise SystemExit(1)
