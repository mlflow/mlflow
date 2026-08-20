"""Verify the real `evaluate_sweep` backend logs the metric keys the sweep UI parses.

Runs an actual sweep (Samraj's `mlflow.genai.evaluate_sweep`) with traced, deterministic
predict functions and a code-based scorer, so no model or credentials are needed, then prints
the exact metric keys logged to the parent run. Those keys are the frontend's contract:
`parseSweepMetrics.ts` infers config/scorer structure from them.

Usage:
    MLFLOW_TRACKING_URI=http://localhost:5000 uv run --no-sync python dev/sweep_contract_check.py
"""

import time

import mlflow
from mlflow.genai.scorers import scorer

EXPERIMENT_NAME = "evaluate_sweep_contract_check"

DATA = [
    {"inputs": {"question": "What is MLflow?"}, "expectations": {"expected": "platform"}},
    {"inputs": {"question": "What is tracing?"}, "expectations": {"expected": "observability"}},
    {"inputs": {"question": "What is a run?"}, "expectations": {"expected": "execution"}},
]

# Per-config canned answers and a fixed delay, so latency percentiles come from real trace
# durations rather than a hardcoded number.
CONFIGS = {
    "fast-model": {"delay": 0.01, "answers": ["platform", "observability", "wrong"]},
    "slow-model": {"delay": 0.05, "answers": ["platform", "observability", "execution"]},
}


@scorer
def exact_match(outputs, expectations) -> bool:
    return expectations.get("expected", "") in str(outputs)


def make_predict_fn(delay: float, answers: list[str]):
    calls = {"n": 0}

    def predict_fn(question: str) -> str:
        # Real work, so the trace records a real execution duration.
        time.sleep(delay)
        answer = answers[calls["n"] % len(answers)]
        calls["n"] += 1
        return f"MLflow is a {answer}"

    return predict_fn


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[exact_match],
        predict_fns={name: make_predict_fn(c["delay"], c["answers"]) for name, c in CONFIGS.items()},
        n_repeats=3,
    )

    parent = mlflow.get_run(result.parent_run_id)
    print(f"\nparent_run_id: {result.parent_run_id}")
    print(f"runType tag:   {parent.data.tags.get('mlflow.runType')!r}")

    print(f"\n--- {len(parent.data.metrics)} metric keys logged to the parent run ---")
    for key in sorted(parent.data.metrics):
        print(f"  {key} = {parent.data.metrics[key]}")

    for name, config in result.configs.items():
        print(f"\n{name}: latency={config.latency} cost={config.cost}")


if __name__ == "__main__":
    main()
