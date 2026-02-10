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
        " FROM experiments e ORDER BY richness DESC LIMIT 3"
    ),
    "dataset": (
        "SELECT i.destination_id AS run_uuid, COUNT(*) AS richness"
        " FROM inputs i WHERE i.destination_type = 'RUN'"
        " GROUP BY i.destination_id ORDER BY richness DESC LIMIT 3"
    ),
    "run": (
        "SELECT r.run_uuid,"
        " (SELECT COUNT(*) FROM params p WHERE p.run_uuid = r.run_uuid)"
        " + (SELECT COUNT(*) FROM metrics m WHERE m.run_uuid = r.run_uuid)"
        " + (SELECT COUNT(*) FROM tags t WHERE t.run_uuid = r.run_uuid)"
        " AS richness"
        " FROM runs r ORDER BY richness DESC LIMIT 3"
    ),
    "trace": (
        "SELECT t.request_id,"
        " (SELECT COUNT(*) FROM trace_tags tt WHERE tt.request_id = t.request_id)"
        " + (SELECT COUNT(*) FROM assessments a WHERE a.trace_id = t.request_id)"
        " AS richness"
        " FROM trace_info t ORDER BY richness DESC LIMIT 3"
    ),
    "logged_model": (
        "SELECT lm.model_id, lm.experiment_id,"
        " (SELECT COUNT(*) FROM logged_model_tags lt"
        "  WHERE lt.model_id = lm.model_id AND lt.experiment_id = lm.experiment_id)"
        " AS richness"
        " FROM logged_models lm ORDER BY richness DESC LIMIT 3"
    ),
    "registered_model": (
        "SELECT rm.name,"
        " (SELECT COUNT(*) FROM model_versions mv WHERE mv.name = rm.name)"
        " + (SELECT COUNT(*) FROM registered_model_tags rt WHERE rt.name = rm.name)"
        " AS richness"
        " FROM registered_models rm ORDER BY richness DESC LIMIT 3"
    ),
    "prompt": (
        "SELECT rm.name,"
        " (SELECT COUNT(*) FROM model_versions mv WHERE mv.name = rm.name)"
        " AS richness"
        " FROM registered_models rm"
        " WHERE rm.name IN"
        "  (SELECT rt.name FROM registered_model_tags rt"
        "   WHERE rt.key = 'mlflow.prompt.is_prompt')"
        " ORDER BY richness DESC LIMIT 3"
    ),
    "model_version": (
        "SELECT mv.name, mv.version,"
        " (SELECT COUNT(*) FROM model_version_tags t"
        "  WHERE t.name = mv.name AND t.version = mv.version)"
        " + (SELECT COUNT(*) FROM registered_model_tags rt"
        "  WHERE rt.name = mv.name)"
        " AS richness"
        " FROM model_versions mv"
        " WHERE mv.name NOT IN"
        "  (SELECT rt2.name FROM registered_model_tags rt2"
        "   WHERE rt2.key = 'mlflow.prompt.is_prompt')"
        " ORDER BY richness DESC LIMIT 3"
    ),
}


def _find_rich(target_uri: str, entity: str) -> list[tuple[object, ...]]:
    engine = create_engine(target_uri)
    query = _RICH_QUERIES.get(entity)
    if not query:
        return []
    with engine.connect() as conn:
        try:
            return conn.execute(text(query)).fetchall()
        except Exception:
            return []


def _check_experiment(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    rows = _find_rich(target_uri, "experiment")
    if not rows:
        _pass("experiment: none found")
        return True

    ok = True
    for row in rows:
        exp_id = str(row[0])
        src_exp = src.get_experiment(exp_id)
        dst_exp = dst.get_experiment(exp_id)

        if src_exp.name != dst_exp.name:
            _fail(f"experiment {exp_id}: name {dst_exp.name!r} != {src_exp.name!r}")
            ok = False
            continue

        if src_exp.lifecycle_stage != dst_exp.lifecycle_stage:
            _fail(
                f"experiment {exp_id}:"
                f" lifecycle_stage {dst_exp.lifecycle_stage!r}"
                f" != {src_exp.lifecycle_stage!r}"
            )
            ok = False
            continue

        src_tags = {k: v for k, v in src_exp.tags.items() if not k.startswith("mlflow.")}
        dst_tags = {k: v for k, v in dst_exp.tags.items() if not k.startswith("mlflow.")}
        if missing := set(src_tags) - set(dst_tags):
            _fail(f"experiment {exp_id}: missing tags {missing}")
            ok = False
            continue

        _pass(
            f"experiment {exp_id}"
            f" (name={dst_exp.name}, lifecycle_stage={dst_exp.lifecycle_stage},"
            f" tags={len(dst_tags)})"
        )
    return ok


def _check_run(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    rows = _find_rich(target_uri, "run")
    if not rows:
        _pass("run: no runs to spot-check")
        return True

    ok = True
    for row in rows:
        run_id = row[0]
        src_run = src.get_run(run_id)
        dst_run = dst.get_run(run_id)
        failed = False

        for field, expected, actual in [
            ("status", src_run.info.status, dst_run.info.status),
            ("lifecycle_stage", src_run.info.lifecycle_stage, dst_run.info.lifecycle_stage),
        ]:
            if expected != actual:
                _fail(f"run {run_id}: {field} {actual!r} != {expected!r}")
                ok = False
                failed = True
        if failed:
            continue

        # Params
        for key, expected_val in src_run.data.params.items():
            actual_val = dst_run.data.params.get(key)
            if actual_val != expected_val:
                _fail(f"run {run_id}: param {key} {actual_val!r} != {expected_val!r}")
                ok = False
                failed = True
                break
        if failed:
            continue

        # Metric keys
        src_metric_keys = set(src_run.data.metrics.keys())
        dst_metric_keys = set(dst_run.data.metrics.keys())
        if src_metric_keys != dst_metric_keys:
            _fail(f"run {run_id}: metric keys {dst_metric_keys} != {src_metric_keys}")
            ok = False
            continue

        # Tags (skip internal mlflow.* tags that may differ between backends)
        src_tags = {k: v for k, v in src_run.data.tags.items() if not k.startswith("mlflow.")}
        dst_tags = {k: v for k, v in dst_run.data.tags.items() if not k.startswith("mlflow.")}
        if missing_tags := set(src_tags) - set(dst_tags):
            _fail(f"run {run_id}: missing tags {missing_tags}")
            ok = False
            continue

        # Dataset inputs
        src_ds = src_run.inputs.dataset_inputs if src_run.inputs else []
        dst_ds = dst_run.inputs.dataset_inputs if dst_run.inputs else []
        if len(src_ds) != len(dst_ds):
            _fail(f"run {run_id}: dataset_inputs {len(dst_ds)} != {len(src_ds)}")
            ok = False
            continue

        _pass(
            f"run {run_id}"
            f" (status={dst_run.info.status},"
            f" params={len(dst_run.data.params)},"
            f" metrics={len(dst_metric_keys)},"
            f" tags={len(dst_tags)},"
            f" datasets={len(dst_ds)})"
        )
    return ok


def _check_dataset(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    rows = _find_rich(target_uri, "dataset")
    if not rows:
        _pass("dataset: none found")
        return True

    ok = True
    for row in rows:
        run_id = row[0]
        src_run = src.get_run(run_id)
        dst_run = dst.get_run(run_id)

        src_ds = src_run.inputs.dataset_inputs if src_run.inputs else []
        dst_ds = dst_run.inputs.dataset_inputs if dst_run.inputs else []

        if len(src_ds) != len(dst_ds):
            _fail(f"dataset on run {run_id}: {len(dst_ds)} inputs != {len(src_ds)}")
            ok = False
            continue

        src_names = sorted(d.dataset.name for d in src_ds)
        dst_names = sorted(d.dataset.name for d in dst_ds)
        if src_names != dst_names:
            _fail(f"dataset on run {run_id}: names {dst_names} != {src_names}")
            ok = False
            continue

        _pass(f"dataset on run {run_id} ({len(dst_ds)} inputs, names={dst_names})")
    return ok


def _find_trace_pair(
    src: MlflowClient, dst: MlflowClient, trace_id: str
) -> tuple[object, object] | None:
    """Find a trace in both source and destination by searching all experiments."""
    src_exps = src.search_experiments(view_type=3)
    for exp in src_exps:
        src_traces = src.search_traces(locations=[exp.experiment_id], max_results=10)
        if src_trace := next((t for t in src_traces if t.info.request_id == trace_id), None):
            dst_traces = dst.search_traces(locations=[exp.experiment_id], max_results=10)
            dst_trace = next((t for t in dst_traces if t.info.request_id == trace_id), None)
            return src_trace, dst_trace
    return None


def _check_trace(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    rows = _find_rich(target_uri, "trace")
    if not rows:
        _pass("trace: none found")
        return True

    ok = True
    for row in rows:
        trace_id = row[0]
        pair = _find_trace_pair(src, dst, trace_id)
        if pair is None:
            _pass(f"trace {trace_id}: not found in source (skipped)")
            continue

        src_trace, dst_trace = pair
        if dst_trace is None:
            _fail(f"trace {trace_id}: missing from DB")
            ok = False
            continue

        if src_trace.info.status != dst_trace.info.status:
            _fail(
                f"trace {trace_id}: status {dst_trace.info.status!r} != {src_trace.info.status!r}"
            )
            ok = False
            continue

        src_tags = src_trace.info.tags
        dst_tags = dst_trace.info.tags
        if missing := set(src_tags) - set(dst_tags):
            _fail(f"trace {trace_id}: missing tags {missing}")
            ok = False
            continue

        _pass(f"trace {trace_id} (status={dst_trace.info.status}, tags={len(dst_tags)})")
    return ok


def _check_assessment(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    rows = _find_rich(target_uri, "trace")
    if not rows:
        _pass("assessment: no traces found")
        return True

    ok = True
    for row in rows:
        trace_id = row[0]
        pair = _find_trace_pair(src, dst, trace_id)
        if pair is None:
            _pass(f"assessment: trace {trace_id} not found in source (skipped)")
            continue

        src_trace, dst_trace = pair
        src_assessments = src_trace.search_assessments(all=True)
        if not src_assessments:
            continue

        if dst_trace is None:
            _fail(f"assessment: trace {trace_id} missing from DB")
            ok = False
            continue

        dst_assessments = dst_trace.search_assessments(all=True)
        src_names = {a.name for a in src_assessments}
        dst_names = {a.name for a in dst_assessments}

        if missing := src_names - dst_names:
            _fail(f"assessment on trace {trace_id}: missing names {missing}")
            ok = False
            continue

        _pass(
            f"assessment on trace {trace_id}"
            f" ({len(dst_assessments)} assessments, names={sorted(dst_names)})"
        )
    return ok


def _check_logged_model(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    rows = _find_rich(target_uri, "logged_model")
    if not rows:
        _pass("logged_model: none found")
        return True

    ok = True
    for row in rows:
        model_id = row[0]
        exp_id = str(row[1])

        src_models = src.search_logged_models(experiment_ids=[exp_id], max_results=10)
        src_model = next((m for m in src_models if m.model_id == model_id), None)
        if src_model is None:
            _pass(f"logged_model {model_id}: not found in source (skipped)")
            continue

        dst_models = dst.search_logged_models(experiment_ids=[exp_id], max_results=10)
        dst_model = next((m for m in dst_models if m.model_id == model_id), None)
        if dst_model is None:
            _fail(f"logged_model {model_id}: missing from DB")
            ok = False
            continue

        if src_model.name != dst_model.name:
            _fail(f"logged_model {model_id}: name {dst_model.name!r} != {src_model.name!r}")
            ok = False
            continue

        if missing_tags := set(src_model.tags) - set(dst_model.tags):
            _fail(f"logged_model {model_id}: missing tags {missing_tags}")
            ok = False
            continue

        _pass(f"logged_model {model_id} (name={dst_model.name}, tags={len(dst_model.tags)})")
    return ok


def _check_registered_model(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    rows = _find_rich(target_uri, "registered_model")
    if not rows:
        _pass("registered_model: none found")
        return True

    ok = True
    for row in rows:
        name = row[0]
        src_models = src.search_registered_models(max_results=10)
        src_model = next((m for m in src_models if m.name == name), None)
        if src_model is None:
            _pass(f"registered_model {name}: not found in source (skipped)")
            continue

        dst_models = dst.search_registered_models(max_results=10)
        dst_model = next((m for m in dst_models if m.name == name), None)
        if dst_model is None:
            _fail(f"registered_model {name}: missing from DB")
            ok = False
            continue

        if src_model.description != dst_model.description:
            _fail(
                f"registered_model {name}:"
                f" description {dst_model.description!r} != {src_model.description!r}"
            )
            ok = False
            continue

        src_versions = src.search_model_versions(f"name='{name}'")
        dst_versions = dst.search_model_versions(f"name='{name}'")
        if len(dst_versions) < len(src_versions):
            _fail(f"registered_model {name}: {len(dst_versions)} versions < {len(src_versions)}")
            ok = False
            continue

        _pass(f"registered_model {name} (versions={len(dst_versions)})")
    return ok


def _check_model_version(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    rows = _find_rich(target_uri, "model_version")
    if not rows:
        _pass("model_version: none found")
        return True

    ok = True
    for row in rows:
        name = row[0]
        version = str(row[1])

        src_mv = src.get_model_version(name, version)
        dst_mv = dst.get_model_version(name, version)

        if src_mv.description != dst_mv.description:
            _fail(
                f"model_version {name}/v{version}:"
                f" description {dst_mv.description!r} != {src_mv.description!r}"
            )
            ok = False
            continue

        if src_mv.status != dst_mv.status:
            _fail(f"model_version {name}/v{version}: status {dst_mv.status!r} != {src_mv.status!r}")
            ok = False
            continue

        if missing := set(src_mv.tags) - set(dst_mv.tags):
            _fail(f"model_version {name}/v{version}: missing tags {missing}")
            ok = False
            continue

        _pass(f"model_version {name}/v{version} (status={dst_mv.status}, tags={len(dst_mv.tags)})")
    return ok


def _check_prompt(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    rows = _find_rich(target_uri, "prompt")
    if not rows:
        _pass("prompt: none found")
        return True

    ok = True
    for row in rows:
        name = row[0]

        src_prompt = src.get_prompt(name)
        if src_prompt is None:
            _pass(f"prompt {name}: not found in source (skipped)")
            continue

        dst_prompt = dst.get_prompt(name)
        if dst_prompt is None:
            _fail(f"prompt {name}: missing from DB")
            ok = False
            continue

        src_pv = src.get_prompt_version(name, 1)
        dst_pv = dst.get_prompt_version(name, 1)
        if src_pv is None:
            _pass(f"prompt {name}: no versions in source (skipped)")
            continue
        if dst_pv is None:
            _fail(f"prompt {name}: version 1 missing from DB")
            ok = False
            continue
        if src_pv.template != dst_pv.template:
            _fail(f"prompt {name}/v1: template {dst_pv.template!r} != {src_pv.template!r}")
            ok = False
            continue

        _pass(f"prompt {name} (template matches)")
    return ok


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
        ok &= _check_dataset(src, dst, target_uri)
        ok &= _check_trace(src, dst, target_uri)
        ok &= _check_assessment(src, dst, target_uri)
        ok &= _check_logged_model(src, dst, target_uri)
        ok &= _check_registered_model(src, dst, target_uri)
        ok &= _check_model_version(src, dst, target_uri)
        ok &= _check_prompt(src, dst, target_uri)

        print()
        if ok:
            print(f"{GREEN}Verification passed{RESET}")
        else:
            print(f"{RED}Verification failed{RESET}")
            raise SystemExit(1)
