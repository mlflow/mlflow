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
        experiments=20,
        runs_per_exp=50,
        datasets_per_run=3,
        traces_per_exp=30,
        assessments_per_trace=5,
        logged_models_per_exp=10,
        registered_models=20,
        prompts=15,
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


def generate_core(cfg: SizeConfig) -> list[ExperimentData]:
    """Create experiments and runs, including edge cases (unicode, NaN, deleted, etc.)."""
    client = MlflowClient()
    result: list[ExperimentData] = []

    for exp_idx in range(cfg.experiments):
        exp_name = f"experiment_{exp_idx}"
        exp_id = client.create_experiment(
            exp_name,
            tags={"team": "ml-infra", "priority": str(exp_idx)},
        )

        run_ids: list[str] = []
        for run_idx in range(cfg.runs_per_exp):
            with mlflow.start_run(
                experiment_id=exp_id,
                tags={"run_index": str(run_idx), "source": "synthetic"},
            ) as run:
                run_ids.append(run.info.run_id)

                mlflow.log_params(
                    {
                        "learning_rate": "0.001",
                        "batch_size": "32",
                        "model_type": f"model_v{run_idx}",
                    }
                )

                mlflow.log_metrics(
                    {
                        "accuracy": 0.85 + run_idx * 0.01,
                        "loss": 0.35 - run_idx * 0.01,
                        "zero_metric": 0.0,
                        "negative_metric": -1.5 + run_idx * 0.1,
                    }
                )
                for step in range(5):
                    mlflow.log_metric("train_loss", 1.0 - step * 0.15, step=step)

                # Artifacts
                mlflow.log_text(f"Run {run_idx} of experiment {exp_idx}", "notes.txt")
                mlflow.log_dict({"lr": 0.001}, "config/params.json")

        result.append(ExperimentData(exp_id, run_ids))

    # ── Edge cases ──

    # NaN / Inf metrics (on first run of first experiment)
    with mlflow.start_run(run_id=result[0].run_ids[0]):
        mlflow.log_metrics(
            {"nan_metric": math.nan, "inf_metric": math.inf, "neg_inf_metric": -math.inf}
        )

    # Unicode experiment name and tag values
    unicode_exp_id = client.create_experiment(
        "実験_テスト_🚀",
        tags={"description": "日本語テスト 🎉", "emoji": "🔬🧪"},
    )
    with mlflow.start_run(experiment_id=unicode_exp_id):
        mlflow.log_params({"unicode_param": "パラメータ値", "long_param": "x" * 8000})

    # Empty run (no metrics/params)
    with mlflow.start_run(experiment_id=result[0].experiment_id):
        pass

    # Deleted experiment with a run
    del_exp_id = client.create_experiment("to_be_deleted")
    with mlflow.start_run(experiment_id=del_exp_id):
        mlflow.log_param("param_in_deleted_exp", "value")
    client.delete_experiment(del_exp_id)

    # Deleted run
    with mlflow.start_run(experiment_id=result[0].experiment_id) as del_run:
        mlflow.log_param("param_in_deleted_run", "value")
    client.delete_run(del_run.info.run_id)

    # Failed run
    failed_run = client.create_run(experiment_id=result[0].experiment_id)
    client.log_param(failed_run.info.run_id, "failed_param", "value")
    client.set_terminated(failed_run.info.run_id, status="FAILED")

    # Killed run
    killed_run = client.create_run(experiment_id=result[0].experiment_id)
    client.log_metric(killed_run.info.run_id, "partial_metric", 0.5)
    client.set_terminated(killed_run.info.run_id, status="KILLED")

    return result


def generate_datasets(cfg: SizeConfig, experiments: list[ExperimentData]) -> None:
    import pandas as pd

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


@mlflow.trace
def _retrieve(query: str) -> list[str]:
    return ["doc1", "doc2"]


@mlflow.trace
def _generate(docs: list[str]) -> str:
    return f"response for {len(docs)} docs"


@mlflow.trace
def _rag_pipeline(query: str) -> str:
    docs = _retrieve(query)
    return _generate(docs)


def generate_traces(cfg: SizeConfig, experiments: list[ExperimentData]) -> list[str]:
    """Returns list of trace IDs."""
    client = MlflowClient()
    trace_ids: list[str] = []

    for exp in experiments:
        mlflow.set_experiment(experiment_id=exp.experiment_id)
        for t_idx in range(cfg.traces_per_exp):
            _rag_pipeline(f"test query {t_idx}")

            trace_id = mlflow.get_last_active_trace_id()
            client.set_trace_tag(trace_id, "trace_source", "synthetic")
            trace_ids.append(trace_id)

    return trace_ids


def generate_assessments(cfg: SizeConfig, trace_ids: list[str]) -> None:
    from mlflow.entities import AssessmentSource, Expectation, Feedback

    human = AssessmentSource(source_type="HUMAN", source_id="test-user")
    ai_judge = AssessmentSource(source_type="AI_JUDGE", source_id="gpt-4o")

    for trace_id in trace_ids:
        for a_idx in range(cfg.assessments_per_trace):
            # Boolean feedback (thumbs up/down)
            mlflow.log_assessment(
                trace_id=trace_id,
                assessment=Feedback(
                    name="correctness",
                    value=a_idx % 2 == 0,
                    source=human,
                    rationale="Looks correct" if a_idx % 2 == 0 else "Has errors",
                ),
            )
            # Numeric feedback (score)
            mlflow.log_assessment(
                trace_id=trace_id,
                assessment=Feedback(
                    name="relevance_score",
                    value=0.6 + a_idx * 0.08,
                    source=ai_judge,
                    rationale=f"Score based on semantic similarity (iteration {a_idx})",
                    metadata={"model": "gpt-4o", "prompt_version": "v1"},
                ),
            )
            # Text feedback
            mlflow.log_assessment(
                trace_id=trace_id,
                assessment=Feedback(
                    name="category",
                    value="good" if a_idx % 3 != 0 else "needs_improvement",
                    source=human,
                ),
            )
            # Expectation (ground truth)
            mlflow.log_assessment(
                trace_id=trace_id,
                assessment=Expectation(
                    name="expected_output",
                    value=f"Expected response for query {a_idx}",
                    source=human,
                ),
            )


def generate_logged_models(cfg: SizeConfig, experiments: list[ExperimentData]) -> list[str]:
    """Returns list of model artifact URIs (runs:/<run_id>/model)."""
    client = MlflowClient()
    model_uris: list[str] = []

    for exp in experiments:
        for m_idx in range(cfg.logged_models_per_exp):
            with mlflow.start_run(experiment_id=exp.experiment_id) as run:
                model_info = mlflow.pyfunc.log_model(
                    name=f"logged_model_{m_idx}",
                    python_model=lambda model_input: model_input,
                    input_example="hello",
                )
                model_uris.append(f"runs:/{run.info.run_id}/model")
            client.set_logged_model_tags(
                model_info.model_id, {"framework": "pytorch", "stage": "dev"}
            )

    return model_uris


def generate_model_registry(cfg: SizeConfig, model_uris: list[str]) -> None:
    client = MlflowClient()

    for rm_idx in range(cfg.registered_models):
        name = f"registered_model_{rm_idx}"
        client.create_registered_model(name, tags={"stage": "staging", "owner": "team-ml"})
        client.update_registered_model(
            name, description=f"Registered model {rm_idx} for testing migration"
        )

        for v_idx in range(1, 3):
            # Use a real logged model URI if available, otherwise fall back to a fake one
            uri_idx = rm_idx * 2 + (v_idx - 1)
            if uri_idx < len(model_uris):
                source = model_uris[uri_idx]
            else:
                source = f"runs:/{uuid.uuid4().hex}/model"
            mv = client.create_model_version(
                name=name,
                source=source,
                tags={"version_note": f"v{v_idx}"},
            )
            client.update_model_version(name, mv.version, description=f"Version {v_idx} of {name}")

        client.set_registered_model_alias(name, "champion", mv.version)


def generate_prompts(cfg: SizeConfig) -> None:
    for p_idx in range(cfg.prompts):
        name = f"prompt_{p_idx}"
        # Version 1
        mlflow.register_prompt(
            name=name,
            template=f"Hello {{{{name}}}}, this is prompt {p_idx}.",
        )
        # Version 2 with updated template
        mlflow.register_prompt(
            name=name,
            template=f"Hi {{{{name}}}}, welcome to prompt {p_idx}. How can I help?",
            commit_message=f"Updated template for prompt {p_idx}",
        )


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

    print("[1/7] Generating experiments, runs, params, metrics, tags, artifacts...")
    experiments = generate_core(cfg)

    if has_feature("datasets"):
        print("[2/7] Generating datasets...")
        generate_datasets(cfg, experiments)
    else:
        print("[2/7] Skipping datasets (not available)")

    trace_ids: list[str] = []
    if has_feature("traces"):
        print("[3/7] Generating traces...")
        trace_ids = generate_traces(cfg, experiments)
    else:
        print("[3/7] Skipping traces (not available)")

    if trace_ids and has_feature("assessments"):
        print("[4/7] Generating assessments...")
        generate_assessments(cfg, trace_ids)
    else:
        print("[4/7] Skipping assessments (not available)")

    model_uris: list[str] = []
    if has_feature("logged_models"):
        print("[5/7] Generating logged models...")
        model_uris = generate_logged_models(cfg, experiments)
    else:
        print("[5/7] Skipping logged models (not available)")

    print("[6/7] Generating model registry...")
    generate_model_registry(cfg, model_uris)

    if has_feature("prompts"):
        print("[7/7] Generating prompts...")
        generate_prompts(cfg)
    else:
        print("[7/7] Skipping prompts (not available)")

    print()
    print(f"Done. Data written to {output}")


if __name__ == "__main__":
    main()
