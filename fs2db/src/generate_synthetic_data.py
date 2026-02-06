# ruff: noqa: T201
"""
Generate synthetic MLflow FileStore data for testing the fs2db migration tool.

Usage:
    uv run --with mlflow==3.5.1 --no-project python -I \
        fs2db/src/generate_synthetic_data.py --output /tmp/fs2db/v3.5.1/ --size small

This script uses the MLflow public API to create realistic on-disk data.
It must only depend on mlflow + stdlib (no local imports).
"""

import argparse
import math
import os
import uuid
from dataclasses import dataclass
from typing import Literal

from packaging.version import Version

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_VERSION = Version(mlflow.__version__)

Size = Literal["small", "full"]


@dataclass
class ExperimentData:
    experiment_id: str
    run_ids: list[str]


# ── Size presets ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SizeConfig:
    experiments: int
    runs_per_exp: int
    datasets_per_run: int
    traces_per_exp: int
    assessments_per_trace: int
    logged_models_per_exp: int
    registered_models: int
    prompts: int


SIZES: dict[Size, SizeConfig] = {
    "small": SizeConfig(
        experiments=2,
        runs_per_exp=2,
        datasets_per_run=1,
        traces_per_exp=1,
        assessments_per_trace=1,
        logged_models_per_exp=1,
        registered_models=1,
        prompts=1,
    ),
    "full": SizeConfig(
        experiments=5,
        runs_per_exp=5,
        datasets_per_run=2,
        traces_per_exp=3,
        assessments_per_trace=2,
        logged_models_per_exp=2,
        registered_models=3,
        prompts=3,
    ),
}

# ── Version detection ─────────────────────────────────────────────────────────


def has_feature(feature: str) -> bool:
    match feature:
        case "datasets":
            return hasattr(mlflow, "data")
        case "traces":
            return MLFLOW_VERSION >= Version("2.14")
        case "assessments":
            # log_assessment(value=...) signature available since 3.6
            return MLFLOW_VERSION >= Version("3.6")
        case "logged_models":
            return hasattr(MlflowClient(), "create_logged_model")
        case "prompts":
            return hasattr(mlflow, "register_prompt")
        case _:
            return False


# ── Generators ────────────────────────────────────────────────────────────────

summary: dict[str, int] = {}


def bump(key: str, n: int = 1) -> None:
    summary[key] = summary.get(key, 0) + n


def generate_core(cfg: SizeConfig) -> list[ExperimentData]:
    """Returns a list of ExperimentData with generated run IDs."""
    client = MlflowClient()
    result: list[ExperimentData] = []

    for exp_idx in range(cfg.experiments):
        exp_name = f"experiment_{exp_idx}"
        exp_id = client.create_experiment(
            exp_name,
            tags={"team": "ml-infra", "priority": str(exp_idx)},
        )
        bump("experiments")
        bump("experiment_tags", 2)

        run_ids: list[str] = []
        for run_idx in range(cfg.runs_per_exp):
            run = client.create_run(
                exp_id,
                tags={"run_index": str(run_idx), "source": "synthetic"},
            )
            rid = run.info.run_id
            run_ids.append(rid)
            bump("runs")
            bump("run_tags", 2)

            client.log_param(rid, "learning_rate", "0.001")
            client.log_param(rid, "batch_size", "32")
            client.log_param(rid, "model_type", f"model_v{run_idx}")
            bump("params", 3)

            client.log_metric(rid, "accuracy", 0.85 + run_idx * 0.01)
            client.log_metric(rid, "loss", 0.35 - run_idx * 0.01)
            bump("metrics", 2)

            # Multi-step metric
            for step in range(5):
                client.log_metric(rid, "train_loss", 1.0 - step * 0.15, step=step)
            bump("metrics (multi-step)", 5)

            client.set_terminated(rid)

        result.append(ExperimentData(exp_id, run_ids))

    return result


def generate_datasets(cfg: SizeConfig, experiments: list[ExperimentData]) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("  [skip] pandas not available, skipping datasets")
        return

    for exp in experiments:
        for rid in exp.run_ids:
            for ds_idx in range(cfg.datasets_per_run):
                df = pd.DataFrame(
                    {"feature": [1, 2, 3], "label": [0, 1, 0]},
                )
                dataset = mlflow.data.from_pandas(
                    df,
                    name=f"dataset_{ds_idx}",
                    targets="label",
                )
                with mlflow.start_run(run_id=rid):
                    mlflow.log_input(dataset, context=f"training_{ds_idx}")
                bump("dataset_inputs")


def generate_traces(cfg: SizeConfig, experiments: list[ExperimentData]) -> list[str]:
    """Returns list of trace IDs."""
    client = MlflowClient()
    trace_ids: list[str] = []

    # end_trace renamed request_id → trace_id in 3.0
    _use_trace_id = MLFLOW_VERSION >= Version("3.0")

    for exp in experiments:
        mlflow.set_experiment(experiment_id=exp.experiment_id)
        for t_idx in range(cfg.traces_per_exp):
            trace_info = client.start_trace(
                name=f"trace_{t_idx}",
                inputs={"query": f"test query {t_idx}"},
                experiment_id=exp.experiment_id,
            )
            trace_id = trace_info.request_id

            if _use_trace_id:
                client.end_trace(
                    trace_id=trace_id,
                    outputs={"response": f"test response {t_idx}"},
                )
            else:
                client.end_trace(
                    request_id=trace_id,
                    outputs={"response": f"test response {t_idx}"},
                )
            bump("traces")

            client.set_trace_tag(trace_id, "trace_source", "synthetic")
            bump("trace_tags")

            trace_ids.append(trace_id)

    return trace_ids


def generate_assessments(cfg: SizeConfig, trace_ids: list[str]) -> None:
    client = MlflowClient()

    for trace_id in trace_ids:
        for a_idx in range(cfg.assessments_per_trace):
            client.log_assessment(
                trace_id=trace_id,
                name=f"quality_{a_idx}",
                source=mlflow.entities.AssessmentSource(source_type="HUMAN", source_id="test-user"),
                value=a_idx % 2 == 0,
            )
            bump("assessments")


def generate_logged_models(cfg: SizeConfig, experiments: list[ExperimentData]) -> None:
    client = MlflowClient()

    for exp in experiments:
        for m_idx in range(cfg.logged_models_per_exp):
            model = client.create_logged_model(
                experiment_id=exp.experiment_id,
                name=f"logged_model_{m_idx}",
            )
            model_id = model.model_id
            bump("logged_models")

            tags = {"framework": "pytorch"}
            if exp.run_ids:
                tags["source_run"] = exp.run_ids[0]
            client.set_logged_model_tags(model_id, tags)
            bump("logged_model_tags", len(tags))


def generate_model_registry(cfg: SizeConfig) -> None:
    client = MlflowClient()

    for rm_idx in range(cfg.registered_models):
        name = f"registered_model_{rm_idx}"
        client.create_registered_model(name, tags={"stage": "staging", "owner": "team-ml"})
        bump("registered_models")
        bump("registered_model_tags", 2)

        for v_idx in range(1, 3):
            mv = client.create_model_version(
                name=name,
                source=f"runs:/{uuid.uuid4().hex}/model",
                tags={"version_note": f"v{v_idx}"},
            )
            bump("model_versions")
            bump("model_version_tags")

        client.set_registered_model_alias(name, "champion", mv.version)
        bump("model_aliases")


def generate_prompts(cfg: SizeConfig) -> None:
    for p_idx in range(cfg.prompts):
        mlflow.register_prompt(
            name=f"prompt_{p_idx}",
            template=f"Hello {{{{name}}}}, this is prompt {p_idx}.",
        )
        bump("prompts")


def generate_edge_cases(experiments: list[ExperimentData]) -> None:
    client = MlflowClient()

    # NaN / Inf metric values
    exp = experiments[0]
    rid = exp.run_ids[0]
    client.log_metric(rid, "nan_metric", math.nan)
    client.log_metric(rid, "inf_metric", math.inf)
    client.log_metric(rid, "neg_inf_metric", -math.inf)
    bump("edge_case_metrics", 3)

    # Unicode experiment name and tag values
    unicode_exp_id = client.create_experiment(
        "実験_テスト_🚀",
        tags={"description": "日本語テスト 🎉", "emoji": "🔬🧪"},
    )
    bump("experiments")
    bump("experiment_tags", 2)

    run = client.create_run(unicode_exp_id)
    client.log_param(run.info.run_id, "unicode_param", "パラメータ値")
    client.set_terminated(run.info.run_id)
    bump("runs")
    bump("params")

    # Max-length param value (8000 chars)
    long_val = "x" * 8000
    client.log_param(run.info.run_id, "long_param", long_val)
    bump("params")

    # Empty run (no metrics/params)
    empty_run = client.create_run(exp.experiment_id)
    client.set_terminated(empty_run.info.run_id)
    bump("runs (empty)")

    # Deleted experiment
    del_exp_id = client.create_experiment("to_be_deleted")
    del_run = client.create_run(del_exp_id)
    client.log_param(del_run.info.run_id, "param_in_deleted_exp", "value")
    client.set_terminated(del_run.info.run_id)
    client.delete_experiment(del_exp_id)
    bump("deleted_experiments")
    bump("runs")

    # Deleted run
    del_run2 = client.create_run(exp.experiment_id)
    client.log_param(del_run2.info.run_id, "param_in_deleted_run", "value")
    client.set_terminated(del_run2.info.run_id)
    client.delete_run(del_run2.info.run_id)
    bump("deleted_runs")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic MLflow FileStore data")
    parser.add_argument(
        "--output",
        required=True,
        help="Root directory for generated mlruns/ data",
    )
    parser.add_argument(
        "--size",
        choices=["small", "full"],
        default="small",
        help="Data size preset (default: small)",
    )
    args = parser.parse_args()

    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=True)

    os.environ["MLFLOW_TRACKING_URI"] = output
    mlflow.set_tracking_uri(output)

    size: Size = args.size
    cfg = SIZES[size]
    print(f"Generating {size} synthetic data in {output}")
    print(f"MLflow version: {mlflow.__version__}")
    print()

    print("[1/8] Generating experiments, runs, params, metrics, tags...")
    experiments = generate_core(cfg)

    if has_feature("datasets"):
        print("[2/8] Generating datasets...")
        generate_datasets(cfg, experiments)
    else:
        print("[2/8] Skipping datasets (not available)")

    trace_ids: list[str] = []
    if has_feature("traces"):
        print("[3/8] Generating traces...")
        trace_ids = generate_traces(cfg, experiments)
    else:
        print("[3/8] Skipping traces (not available)")

    if trace_ids and has_feature("assessments"):
        print("[4/8] Generating assessments...")
        generate_assessments(cfg, trace_ids)
    else:
        print("[4/8] Skipping assessments (not available)")

    if has_feature("logged_models"):
        print("[5/8] Generating logged models...")
        generate_logged_models(cfg, experiments)
    else:
        print("[5/8] Skipping logged models (not available)")

    print("[6/8] Generating model registry...")
    generate_model_registry(cfg)

    if has_feature("prompts"):
        print("[7/8] Generating prompts...")
        generate_prompts(cfg)
    else:
        print("[7/8] Skipping prompts (not available)")

    print("[8/8] Generating edge cases...")
    generate_edge_cases(experiments)

    print()
    print("=" * 50)
    print("Summary:")
    print("=" * 50)
    for key, count in sorted(summary.items()):
        print(f"  {key}: {count}")
    print("=" * 50)
    print(f"Done. Data written to {output}")


if __name__ == "__main__":
    main()
