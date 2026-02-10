# ruff: noqa: T201
"""
Verify a fs2db migration by comparing MLflow public API results from source
(FileStore) and target (DB) backends.

Strategy:
  1. Row counts — use SQL on the target DB for accurate totals.
  2. Spot checks — sample individual entities via MLflow public API from
     both source (FileStore) and target (DB) and compare fields.
"""

import warnings
from pathlib import Path

from sqlalchemy import create_engine, text

from mlflow.store.fs2db import _resolve_mlruns
from mlflow.tracking import MlflowClient

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def _pass(msg: str) -> None:
    print(f"  {GREEN}PASS{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET} {msg}")


# ---------------------------------------------------------------------------
# 1. Row-count verification (SQL)
# ---------------------------------------------------------------------------

_COUNT_QUERIES: dict[str, str] = {
    "experiments": "SELECT COUNT(*) FROM experiments",
    "runs": "SELECT COUNT(*) FROM runs",
    "params": "SELECT COUNT(*) FROM params",
    "tags": "SELECT COUNT(*) FROM tags",
    "metrics": "SELECT COUNT(*) FROM metrics",
    "datasets": "SELECT COUNT(*) FROM datasets",
    "traces": "SELECT COUNT(*) FROM trace_info",
    "assessments": "SELECT COUNT(*) FROM assessments",
    "logged_models": "SELECT COUNT(*) FROM logged_models",
    "registered_models": "SELECT COUNT(*) FROM registered_models",
    "model_versions": "SELECT COUNT(*) FROM model_versions",
}


def _check_row_counts(target_uri: str) -> bool:
    ok = True
    engine = create_engine(target_uri)
    with engine.connect() as conn:
        for entity, query in _COUNT_QUERIES.items():
            try:
                count = conn.execute(text(query)).scalar()
            except Exception:
                count = 0
            if count > 0:
                _pass(f"{entity}: {count} rows")
            else:
                _pass(f"{entity}: 0 rows (skipped)")
    return ok


# ---------------------------------------------------------------------------
# 2. Spot checks (public API)
# ---------------------------------------------------------------------------

# SQL queries to find the richest entity for each type.
_RICH_QUERIES: dict[str, str] = {
    "experiment": (
        "SELECT e.experiment_id,"
        " (SELECT COUNT(*) FROM experiment_tags et"
        "  WHERE et.experiment_id = e.experiment_id)"
        " + (SELECT COUNT(*) FROM runs r WHERE r.experiment_id = e.experiment_id)"
        " AS richness"
        " FROM experiments e ORDER BY richness DESC LIMIT 1"
    ),
    "run": (
        "SELECT r.run_uuid,"
        " (SELECT COUNT(*) FROM params p WHERE p.run_uuid = r.run_uuid)"
        " + (SELECT COUNT(*) FROM metrics m WHERE m.run_uuid = r.run_uuid)"
        " + (SELECT COUNT(*) FROM tags t WHERE t.run_uuid = r.run_uuid)"
        " AS richness"
        " FROM runs r ORDER BY richness DESC LIMIT 1"
    ),
    "trace": (
        "SELECT t.request_id,"
        " (SELECT COUNT(*) FROM trace_tags tt WHERE tt.request_id = t.request_id)"
        " + (SELECT COUNT(*) FROM assessments a WHERE a.trace_id = t.request_id)"
        " AS richness"
        " FROM trace_info t ORDER BY richness DESC LIMIT 1"
    ),
    "logged_model": (
        "SELECT lm.model_id, lm.experiment_id,"
        " (SELECT COUNT(*) FROM logged_model_tags lt"
        "  WHERE lt.model_id = lm.model_id AND lt.experiment_id = lm.experiment_id)"
        " AS richness"
        " FROM logged_models lm ORDER BY richness DESC LIMIT 1"
    ),
    "registered_model": (
        "SELECT rm.name,"
        " (SELECT COUNT(*) FROM model_versions mv WHERE mv.name = rm.name)"
        " + (SELECT COUNT(*) FROM registered_model_tags rt WHERE rt.name = rm.name)"
        " AS richness"
        " FROM registered_models rm ORDER BY richness DESC LIMIT 1"
    ),
}


def _find_rich(target_uri: str, entity: str) -> tuple[object, ...] | None:
    engine = create_engine(target_uri)
    query = _RICH_QUERIES.get(entity)
    if not query:
        return None
    with engine.connect() as conn:
        try:
            return conn.execute(text(query)).first()
        except Exception:
            return None


def _check_experiment(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    row = _find_rich(target_uri, "experiment")
    if not row:
        _pass("experiment: none found")
        return True
    exp_id = str(row[0])

    src_exp = src.get_experiment(exp_id)
    dst_exp = dst.get_experiment(exp_id)

    if src_exp.name != dst_exp.name:
        _fail(f"experiment {exp_id}: name {dst_exp.name!r} != {src_exp.name!r}")
        return False

    if src_exp.lifecycle_stage != dst_exp.lifecycle_stage:
        _fail(
            f"experiment {exp_id}:"
            f" lifecycle_stage {dst_exp.lifecycle_stage!r} != {src_exp.lifecycle_stage!r}"
        )
        return False

    src_tags = {k: v for k, v in src_exp.tags.items() if not k.startswith("mlflow.")}
    dst_tags = {k: v for k, v in dst_exp.tags.items() if not k.startswith("mlflow.")}
    if missing := set(src_tags) - set(dst_tags):
        _fail(f"experiment {exp_id}: missing tags {missing}")
        return False

    _pass(
        f"experiment {exp_id}"
        f" (name={dst_exp.name}, lifecycle_stage={dst_exp.lifecycle_stage},"
        f" tags={len(dst_tags)})"
    )
    return True


def _check_run(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    row = _find_rich(target_uri, "run")
    if not row:
        _pass("run: no runs to spot-check")
        return True
    run_id = row[0]

    src_run = src.get_run(run_id)
    dst_run = dst.get_run(run_id)

    checks = [
        ("status", src_run.info.status, dst_run.info.status),
        ("lifecycle_stage", src_run.info.lifecycle_stage, dst_run.info.lifecycle_stage),
    ]
    for field, expected, actual in checks:
        if expected != actual:
            _fail(f"run {run_id}: {field} {actual!r} != {expected!r}")
            return False

    # Params
    for key, expected_val in src_run.data.params.items():
        actual_val = dst_run.data.params.get(key)
        if actual_val != expected_val:
            _fail(f"run {run_id}: param {key} {actual_val!r} != {expected_val!r}")
            return False

    # Metric keys
    src_metric_keys = set(src_run.data.metrics.keys())
    dst_metric_keys = set(dst_run.data.metrics.keys())
    if src_metric_keys != dst_metric_keys:
        _fail(f"run {run_id}: metric keys {dst_metric_keys} != {src_metric_keys}")
        return False

    # Tags (skip internal mlflow.* tags that may differ between backends)
    src_tags = {k: v for k, v in src_run.data.tags.items() if not k.startswith("mlflow.")}
    dst_tags = {k: v for k, v in dst_run.data.tags.items() if not k.startswith("mlflow.")}
    if missing_tags := set(src_tags) - set(dst_tags):
        _fail(f"run {run_id}: missing tags {missing_tags}")
        return False

    # Dataset inputs
    src_ds = src_run.inputs.dataset_inputs if src_run.inputs else []
    dst_ds = dst_run.inputs.dataset_inputs if dst_run.inputs else []
    if len(src_ds) != len(dst_ds):
        _fail(f"run {run_id}: dataset_inputs {len(dst_ds)} != {len(src_ds)}")
        return False

    _pass(
        f"run {run_id}"
        f" (status={dst_run.info.status},"
        f" params={len(dst_run.data.params)},"
        f" metrics={len(dst_metric_keys)},"
        f" tags={len(dst_tags)},"
        f" datasets={len(dst_ds)})"
    )
    return True


def _check_trace(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    row = _find_rich(target_uri, "trace")
    if not row:
        _pass("trace: none found")
        return True
    trace_id = row[0]

    # Find the experiment this trace belongs to, then search from both sides
    src_exps = src.search_experiments(view_type=3)
    for exp in src_exps:
        src_traces = src.search_traces(locations=[exp.experiment_id], max_results=10)
        src_trace = next((t for t in src_traces if t.info.request_id == trace_id), None)
        if src_trace:
            break
    else:
        _pass(f"trace {trace_id}: not found in source (skipped)")
        return True

    dst_traces = dst.search_traces(locations=[exp.experiment_id], max_results=10)
    dst_trace = next((t for t in dst_traces if t.info.request_id == trace_id), None)
    if dst_trace is None:
        _fail(f"trace {trace_id}: missing from DB")
        return False

    if src_trace.info.status != dst_trace.info.status:
        _fail(f"trace {trace_id}: status {dst_trace.info.status!r} != {src_trace.info.status!r}")
        return False

    src_tags = src_trace.info.tags
    dst_tags = dst_trace.info.tags
    if missing := set(src_tags) - set(dst_tags):
        _fail(f"trace {trace_id}: missing tags {missing}")
        return False

    _pass(f"trace {trace_id} (status={dst_trace.info.status}, tags={len(dst_tags)})")
    return True


def _check_assessment(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    # The "trace" rich query already picks the trace with the most assessments
    row = _find_rich(target_uri, "trace")
    if not row:
        _pass("assessment: no traces found")
        return True
    trace_id = row[0]

    src_exps = src.search_experiments(view_type=3)
    for exp in src_exps:
        src_traces = src.search_traces(locations=[exp.experiment_id], max_results=10)
        src_trace = next((t for t in src_traces if t.info.request_id == trace_id), None)
        if src_trace:
            break
    else:
        _pass(f"assessment: trace {trace_id} not found in source (skipped)")
        return True

    src_assessments = src_trace.search_assessments(all=True)
    if not src_assessments:
        _pass("assessment: none found on trace")
        return True

    dst_traces = dst.search_traces(locations=[exp.experiment_id], max_results=10)
    dst_trace = next((t for t in dst_traces if t.info.request_id == trace_id), None)
    if dst_trace is None:
        _fail(f"assessment: trace {trace_id} missing from DB")
        return False

    dst_assessments = dst_trace.search_assessments(all=True)
    src_names = {a.name for a in src_assessments}
    dst_names = {a.name for a in dst_assessments}

    if missing := src_names - dst_names:
        _fail(f"assessment on trace {trace_id}: missing names {missing}")
        return False

    _pass(
        f"assessment on trace {trace_id}"
        f" ({len(dst_assessments)} assessments, names={sorted(dst_names)})"
    )
    return True


def _check_logged_model(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    row = _find_rich(target_uri, "logged_model")
    if not row:
        _pass("logged_model: none found")
        return True
    model_id = row[0]
    exp_id = str(row[1])

    src_models = src.search_logged_models(experiment_ids=[exp_id], max_results=10)
    src_model = next((m for m in src_models if m.model_id == model_id), None)
    if src_model is None:
        _pass(f"logged_model {model_id}: not found in source (skipped)")
        return True

    dst_models = dst.search_logged_models(experiment_ids=[exp_id], max_results=10)
    dst_model = next((m for m in dst_models if m.model_id == model_id), None)
    if dst_model is None:
        _fail(f"logged_model {model_id}: missing from DB")
        return False

    if src_model.name != dst_model.name:
        _fail(f"logged_model {model_id}: name {dst_model.name!r} != {src_model.name!r}")
        return False

    if missing_tags := set(src_model.tags) - set(dst_model.tags):
        _fail(f"logged_model {model_id}: missing tags {missing_tags}")
        return False

    _pass(f"logged_model {model_id} (name={dst_model.name}, tags={len(dst_model.tags)})")
    return True


def _check_registered_model(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    row = _find_rich(target_uri, "registered_model")
    if not row:
        _pass("registered_model: none found")
        return True
    name = row[0]

    src_models = src.search_registered_models(max_results=10)
    src_model = next((m for m in src_models if m.name == name), None)
    if src_model is None:
        _pass(f"registered_model {name}: not found in source (skipped)")
        return True

    dst_models = dst.search_registered_models(max_results=10)
    dst_model = next((m for m in dst_models if m.name == name), None)
    if dst_model is None:
        _fail(f"registered_model {name}: missing from DB")
        return False

    if src_model.description != dst_model.description:
        _fail(
            f"registered_model {name}:"
            f" description {dst_model.description!r} != {src_model.description!r}"
        )
        return False

    src_versions = src.search_model_versions(f"name='{name}'")
    dst_versions = dst.search_model_versions(f"name='{name}'")
    if len(dst_versions) < len(src_versions):
        _fail(f"registered_model {name}: {len(dst_versions)} versions < {len(src_versions)}")
        return False

    _pass(f"registered_model {name} (versions={len(dst_versions)})")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def verify_migration(source: Path, target_uri: str) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*filesystem.*deprecated.*", category=FutureWarning
        )

        mlruns = _resolve_mlruns(source)
        src = MlflowClient(tracking_uri=str(mlruns))
        dst = MlflowClient(tracking_uri=target_uri)

        print()
        print("Row counts:")
        ok = _check_row_counts(target_uri)

        print()
        print("Spot checks:")
        ok &= _check_experiment(src, dst, target_uri)
        ok &= _check_run(src, dst, target_uri)
        ok &= _check_trace(src, dst, target_uri)
        ok &= _check_assessment(src, dst, target_uri)
        ok &= _check_logged_model(src, dst, target_uri)
        ok &= _check_registered_model(src, dst, target_uri)

        print()
        if ok:
            print(f"{GREEN}Verification passed{RESET}")
        else:
            print(f"{RED}Verification failed{RESET}")
            raise SystemExit(1)
