"""Run a real `evaluate_sweep` whose traces carry token usage, so cost is computed end to end.

Complements `dev/sweep_contract_check.py`, which leaves cost empty because its predict functions
report no token usage. Here each predict function sets the model and token-usage span attributes
that MLflow reads to price a trace (model name x token counts x LiteLLM pricing), so the
`{config}/cost_per_request_usd` metric is produced by the real cost pipeline rather than a fixture.

No model is called and no credentials are needed: the token counts are canned, but everything
downstream of them — pricing, trace aggregation, metric logging, and the UI's parsing — is real.

Usage:
    MLFLOW_TRACKING_URI=http://localhost:5000 uv run --no-sync python dev/sweep_cost_check.py
"""

import time

import mlflow
from mlflow.genai.scorers import scorer
from mlflow.tracing.constant import SpanAttributeKey

EXPERIMENT_NAME = "evaluate_sweep_cost_check"

DATA = [
    {"inputs": {"question": "What is MLflow?"}, "expectations": {"expected": "platform"}},
    {"inputs": {"question": "What is tracing?"}, "expectations": {"expected": "observability"}},
    {"inputs": {"question": "What is a run?"}, "expectations": {"expected": "execution"}},
]

# Real model names, so LiteLLM has pricing for them. Token counts and latency are canned, but the
# cost MLflow derives from them is not.
CONFIGS = {
    "gpt-4o": {
        "model": "gpt-4o",
        "tokens": (1200, 180),
        "delay": 0.05,
        "answers": ["platform", "observability", "execution"],
    },
    "gpt-4o-mini": {
        "model": "gpt-4o-mini",
        "tokens": (1200, 190),
        "delay": 0.02,
        "answers": ["platform", "observability", "wrong"],
    },
    "gpt-3.5-turbo": {
        "model": "gpt-3.5-turbo",
        "tokens": (1250, 210),
        "delay": 0.01,
        "answers": ["platform", "wrong", "wrong"],
    },
}


@scorer
def exact_match(outputs, expectations) -> bool:
    return expectations.get("expected", "") in str(outputs)


def make_predict_fn(model: str, tokens: tuple[int, int], delay: float, answers: list[str]):
    calls = {"n": 0}
    input_tokens, output_tokens = tokens

    def predict_fn(question: str) -> str:
        time.sleep(delay)
        answer = answers[calls["n"] % len(answers)]
        calls["n"] += 1

        # The attributes MLflow's cost pipeline reads. Setting them on the active span is what a
        # real autologged LLM integration does after a completion returns.
        span = mlflow.get_current_active_span()
        if span is not None:
            span.set_attribute(SpanAttributeKey.MODEL, model)
            span.set_attribute(SpanAttributeKey.MODEL_PROVIDER, "openai")
            span.set_attribute(
                SpanAttributeKey.CHAT_USAGE,
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            )
        return f"MLflow is a {answer}"

    return predict_fn


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    result = mlflow.genai.evaluate_sweep(
        data=DATA,
        scorers=[exact_match],
        predict_fns={
            name: make_predict_fn(c["model"], c["tokens"], c["delay"], c["answers"])
            for name, c in CONFIGS.items()
        },
        n_repeats=3,
    )

    parent = mlflow.get_run(result.parent_run_id)
    print(f"\nparent_run_id: {result.parent_run_id}")

    print("\n--- cost and latency, aggregated from traces ---")
    for name, config in result.configs.items():
        print(f"  {name}: cost={config.cost} latency_p50={config.latency.p50 if config.latency else None}")

    cost_keys = sorted(k for k in parent.data.metrics if "cost" in k)
    print(f"\n--- {len(cost_keys)} cost metric(s) logged to the parent run ---")
    for key in cost_keys:
        print(f"  {key} = {parent.data.metrics[key]}")


if __name__ == "__main__":
    main()
