# ruff: noqa: T201
"""
Verify a fs2db migration by comparing MLflow public API results from source
(FileStore) and target (DB) backends.
"""

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


def _check_experiments(src: MlflowClient, dst: MlflowClient) -> bool:
    ok = True
    src_exps = src.search_experiments(view_type=3)  # ALL
    dst_exps = dst.search_experiments(view_type=3)
    src_by_id = {e.experiment_id: e for e in src_exps}
    dst_by_id = {e.experiment_id: e for e in dst_exps}

    if len(dst_by_id) < len(src_by_id):
        _fail(f"experiments: {len(dst_by_id)} < {len(src_by_id)}")
        ok = False
    else:
        _pass(f"experiments: {len(dst_by_id)} (source: {len(src_by_id)})")

    for exp_id, src_exp in src_by_id.items():
        dst_exp = dst_by_id.get(exp_id)
        if dst_exp is None:
            _fail(f"experiment {exp_id}: missing from DB")
            ok = False
            continue
        if src_exp.name != dst_exp.name:
            _fail(f"experiment {exp_id}: name {dst_exp.name!r} != {src_exp.name!r}")
            ok = False
        elif src_exp.lifecycle_stage != dst_exp.lifecycle_stage:
            _fail(
                f"experiment {exp_id}: lifecycle_stage"
                f" {dst_exp.lifecycle_stage!r} != {src_exp.lifecycle_stage!r}"
            )
            ok = False
        break  # spot-check first
    return ok


def _find_rich_run_id(target_uri: str) -> str | None:
    """Find a run with the most params+metrics for a meaningful comparison."""
    engine = create_engine(target_uri)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT r.run_uuid,"
                " (SELECT COUNT(*) FROM params p WHERE p.run_uuid = r.run_uuid)"
                " + (SELECT COUNT(*) FROM metrics m WHERE m.run_uuid = r.run_uuid) AS richness"
                " FROM runs r ORDER BY richness DESC LIMIT 1"
            )
        ).first()
        return row[0] if row else None


def _check_runs(src: MlflowClient, dst: MlflowClient, target_uri: str) -> bool:
    ok = True
    run_id = _find_rich_run_id(target_uri)
    if not run_id:
        _pass("runs: no runs to check")
        return ok

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

    # Compare params
    for key, expected_val in src_run.data.params.items():
        actual_val = dst_run.data.params.get(key)
        if actual_val != expected_val:
            _fail(f"run {run_id}: param {key} {actual_val!r} != {expected_val!r}")
            return False

    # Compare metric keys
    src_metric_keys = set(src_run.data.metrics.keys())
    dst_metric_keys = set(dst_run.data.metrics.keys())
    if src_metric_keys != dst_metric_keys:
        _fail(f"run {run_id}: metric keys {dst_metric_keys} != {src_metric_keys}")
        return False

    # Compare tags (skip internal mlflow.* tags that may differ between backends)
    src_tags = {k: v for k, v in src_run.data.tags.items() if not k.startswith("mlflow.")}
    dst_tags = {k: v for k, v in dst_run.data.tags.items() if not k.startswith("mlflow.")}
    if missing_tags := set(src_tags) - set(dst_tags):
        _fail(f"run {run_id}: missing tags {missing_tags}")
        return False

    # Compare dataset inputs
    src_ds = src_run.inputs.dataset_inputs if src_run.inputs else []
    dst_ds = dst_run.inputs.dataset_inputs if dst_run.inputs else []
    if len(src_ds) != len(dst_ds):
        _fail(f"run {run_id}: dataset_inputs {len(dst_ds)} != {len(src_ds)}")
        ok = False
    elif src_ds:
        src_ds_names = sorted(d.dataset.name for d in src_ds)
        dst_ds_names = sorted(d.dataset.name for d in dst_ds)
        if src_ds_names != dst_ds_names:
            _fail(f"run {run_id}: dataset names {dst_ds_names} != {src_ds_names}")
            ok = False

    _pass(
        f"run {run_id}"
        f" (status={dst_run.info.status},"
        f" params={len(dst_run.data.params)},"
        f" metrics={len(dst_metric_keys)},"
        f" tags={len(dst_tags)},"
        f" datasets={len(dst_ds)})"
    )
    return ok


def _check_traces(src: MlflowClient, dst: MlflowClient) -> bool:
    ok = True
    src_exps = src.search_experiments(view_type=3)
    for exp in src_exps:
        src_traces = src.search_traces(experiment_ids=[exp.experiment_id], max_results=1)
        if not src_traces:
            continue

        src_trace = src_traces[0]
        dst_traces = dst.search_traces(experiment_ids=[exp.experiment_id], max_results=5000)
        dst_by_id = {t.info.request_id: t for t in dst_traces}
        dst_trace = dst_by_id.get(src_trace.info.request_id)

        if dst_trace is None:
            _fail(f"trace {src_trace.info.request_id}: missing from DB")
            return False

        if src_trace.info.status != dst_trace.info.status:
            _fail(
                f"trace {src_trace.info.request_id}:"
                f" status {dst_trace.info.status!r} != {src_trace.info.status!r}"
            )
            ok = False
        else:
            src_tags = src_trace.info.tags
            dst_tags = dst_trace.info.tags
            if missing := set(src_tags) - set(dst_tags):
                _fail(f"trace {src_trace.info.request_id}: missing tags {missing}")
                ok = False
            else:
                _pass(
                    f"trace {src_trace.info.request_id}"
                    f" (status={dst_trace.info.status}, tags={len(dst_tags)})"
                )
        return ok
    _pass("traces: none found")
    return ok


def _check_assessments(src: MlflowClient, dst: MlflowClient) -> bool:
    src_exps = src.search_experiments(view_type=3)
    for exp in src_exps:
        src_traces = src.search_traces(experiment_ids=[exp.experiment_id], max_results=1)
        if not src_traces:
            continue

        src_trace = src_traces[0]
        src_assessments = src_trace.search_assessments(all=True)
        if not src_assessments:
            continue

        dst_traces = dst.search_traces(experiment_ids=[exp.experiment_id], max_results=5000)
        dst_trace = next(
            (t for t in dst_traces if t.info.request_id == src_trace.info.request_id), None
        )
        if dst_trace is None:
            _fail(f"assessments: trace {src_trace.info.request_id} missing from DB")
            return False

        dst_assessments = dst_trace.search_assessments(all=True)

        # Compare by name
        src_by_name = {}
        for a in src_assessments:
            src_by_name.setdefault(a.name, []).append(a)
        dst_by_name = {}
        for a in dst_assessments:
            dst_by_name.setdefault(a.name, []).append(a)

        if missing := set(src_by_name) - set(dst_by_name):
            _fail(f"assessments on trace {src_trace.info.request_id}: missing names {missing}")
            return False

        _pass(
            f"assessments on trace {src_trace.info.request_id}"
            f" ({len(dst_assessments)} assessments, names={sorted(dst_by_name.keys())})"
        )
        return True

    _pass("assessments: none found")
    return True


def _check_logged_models(src: MlflowClient, dst: MlflowClient) -> bool:
    src_exps = src.search_experiments(view_type=3)
    exp_ids = [e.experiment_id for e in src_exps]
    if not exp_ids:
        _pass("logged_models: no experiments")
        return True

    src_models = src.search_logged_models(experiment_ids=exp_ids)
    dst_models = dst.search_logged_models(experiment_ids=exp_ids)

    if len(dst_models) < len(src_models):
        _fail(f"logged_models: {len(dst_models)} < {len(src_models)}")
        return False

    if not src_models:
        _pass("logged_models: none found")
        return True

    # Spot-check first model
    src_by_id = {m.model_id: m for m in src_models}
    dst_by_id = {m.model_id: m for m in dst_models}

    for model_id, src_model in src_by_id.items():
        dst_model = dst_by_id.get(model_id)
        if dst_model is None:
            _fail(f"logged_model {model_id}: missing from DB")
            return False

        if src_model.name != dst_model.name:
            _fail(f"logged_model {model_id}: name {dst_model.name!r} != {src_model.name!r}")
            return False

        # Compare tags
        if missing_tags := set(src_model.tags) - set(dst_model.tags):
            _fail(f"logged_model {model_id}: missing tags {missing_tags}")
            return False

        _pass(
            f"logged_models: {len(dst_models)} (source: {len(src_models)}),"
            f" spot-check {model_id} (name={dst_model.name}, tags={len(dst_model.tags)})"
        )
        return True

    _pass(f"logged_models: {len(dst_models)} (source: {len(src_models)})")
    return True


def _check_registered_models(src: MlflowClient, dst: MlflowClient) -> bool:
    ok = True
    src_models = src.search_registered_models()
    dst_models = dst.search_registered_models()
    src_by_name = {m.name: m for m in src_models}
    dst_by_name = {m.name: m for m in dst_models}

    if len(dst_by_name) < len(src_by_name):
        _fail(f"registered_models: {len(dst_by_name)} < {len(src_by_name)}")
        ok = False
    else:
        _pass(f"registered_models: {len(dst_by_name)} (source: {len(src_by_name)})")

    for name, src_model in src_by_name.items():
        dst_model = dst_by_name.get(name)
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
        else:
            # Check version count
            src_versions = src.search_model_versions(f"name='{name}'")
            dst_versions = dst.search_model_versions(f"name='{name}'")
            if len(dst_versions) < len(src_versions):
                _fail(
                    f"registered_model {name}: {len(dst_versions)} versions < {len(src_versions)}"
                )
                ok = False
            else:
                _pass(f"registered_model {name} (versions={len(dst_versions)})")
        break  # spot-check first
    return ok


def verify_migration(source: Path, target_uri: str) -> None:
    mlruns = _resolve_mlruns(source)
    src = MlflowClient(tracking_uri=str(mlruns))
    dst = MlflowClient(tracking_uri=target_uri)

    print()
    print("Verification:")
    ok = True
    ok &= _check_experiments(src, dst)
    ok &= _check_runs(src, dst, target_uri)
    ok &= _check_traces(src, dst)
    ok &= _check_assessments(src, dst)
    ok &= _check_logged_models(src, dst)
    ok &= _check_registered_models(src, dst)

    print()
    if ok:
        print(f"{GREEN}Verification passed{RESET}")
    else:
        print(f"{RED}Verification failed{RESET}")
        raise SystemExit(1)
