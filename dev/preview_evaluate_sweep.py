"""Create a fake `evaluate_sweep` parent run so the Evaluation sweep UI tab can be previewed.

Logs the same tag and summary metrics that :func:`mlflow.genai.evaluate_sweep` flattens onto its
parent run, without calling any model or scorer. Also creates a plain run, to check that the tab
only shows up on the sweep run.

Usage:
    MLFLOW_TRACKING_URI=http://localhost:5000 uv run --no-sync python dev/preview_evaluate_sweep.py
"""

import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE

# Must match MLFLOW_RUN_TYPE_GENAI_EVALUATE_SWEEP in mlflow/genai/evaluation/sweep.py.
RUN_TYPE_GENAI_EVALUATE_SWEEP = "genai_evaluate_sweep"

EXPERIMENT_NAME = "evaluate_sweep_ui_preview"

# {config: {scorer: (mean, ci_low, ci_high, std)}}, plus latency percentiles and cost per request.
CONFIGS = {
    "gpt-4o": {
        "scorers": {
            "correctness": (0.91, 0.88, 0.94, 0.021),
            "safety": (0.99, 0.97, 1.0, 0.008),
        },
        "latency": {"p50": 412.0, "p90": 690.0, "p95": 780.0, "p99": 910.0},
        "cost_per_request_usd": 0.0128,
    },
    "claude-sonnet": {
        "scorers": {
            # Overlaps gpt-4o's correctness interval, so both are tagged "Best".
            "correctness": (0.89, 0.86, 0.93, 0.024),
            "safety": (0.94, 0.91, 0.97, 0.014),
        },
        "latency": {"p50": 355.0, "p90": 601.0, "p95": 702.0, "p99": 845.0},
        "cost_per_request_usd": 0.0091,
    },
    "llama-3-8b": {
        "scorers": {
            "correctness": (0.62, 0.57, 0.67, 0.038),
            "safety": (0.88, 0.84, 0.92, 0.019),
        },
        "latency": {"p50": 128.0, "p90": 210.0, "p95": 249.0, "p99": 302.0},
        "cost_per_request_usd": 0.0004,
    },
}


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="sweep-preview") as parent:
        mlflow.set_tag(MLFLOW_RUN_TYPE, RUN_TYPE_GENAI_EVALUATE_SWEEP)

        metrics = {}
        for config, spec in CONFIGS.items():
            for scorer, (mean, ci_low, ci_high, std) in spec["scorers"].items():
                metrics[f"{config}/{scorer}/mean"] = mean
                metrics[f"{config}/{scorer}/ci_low"] = ci_low
                metrics[f"{config}/{scorer}/ci_high"] = ci_high
                metrics[f"{config}/{scorer}/std"] = std
            for percentile, value in spec["latency"].items():
                metrics[f"{config}/latency_{percentile}_ms"] = value
            metrics[f"{config}/cost_per_request_usd"] = spec["cost_per_request_usd"]

        mlflow.log_metrics(metrics)
        print(f"Sweep parent run: {parent.info.run_id}")

    with mlflow.start_run(run_name="plain-run") as plain:
        mlflow.log_metric("accuracy", 0.87)
        print(f"Plain run (tab should be hidden): {plain.info.run_id}")

    print(f"\nOpen the '{EXPERIMENT_NAME}' experiment and select the sweep parent run.")


if __name__ == "__main__":
    main()
