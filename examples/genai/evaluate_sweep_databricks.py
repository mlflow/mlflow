"""Sample: sweep multiple models with ``mlflow.genai.evaluate_sweep`` on Databricks.

Run this in a Databricks notebook or against a Databricks workspace
(``mlflow.set_tracking_uri("databricks")``). It shows the intended pattern:

1. An evaluation dataset of test cases (inputs + expectations).
2. A single ``make_agent(model)`` factory that builds a traced ``predict_fn`` —
   so swapping models is just changing a string.
3. A sweep across those models with repeats, producing per-scorer confidence
   intervals and a quality-vs-latency comparison.

Assumes access to a Databricks workspace and its AI Gateway (the built-in
scorers are LLM judges, so they call a model too).
"""

import os

from openai import OpenAI

import mlflow
from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery, Safety

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
# Databricks workspace + AI Gateway base URL. In a Databricks notebook,
# DATABRICKS_TOKEN is available for you; otherwise set it in your environment.
DATABRICKS_HOST = "https://e2-dogfood.staging.cloud.databricks.com"
AI_GATEWAY_BASE_URL = f"{DATABRICKS_HOST}/ai-gateway/mlflow/v1"

# Models to compare, addressed through the AI Gateway. Swap in whatever your
# workspace serves — the sweep keys each config off these names.
MODELS = [
    "system.ai.claude-sonnet-4-6",
    "system.ai.llama-4-maverick",
    "system.ai.gpt-oss-120b",
]

# Point MLflow at the workspace. In a Databricks notebook this is the default.
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/me@example.com/evaluate-sweep-demo")


# ---------------------------------------------------------------------------
# 1. Evaluation dataset — a list of test cases
# ---------------------------------------------------------------------------
# Each case has `inputs` (passed to the agent as kwargs) and `expectations`
# (ground truth the scorers compare against).
EVAL_DATA = [
    {
        "inputs": {"question": "What is MLflow Tracking used for?"},
        "expectations": {
            "expected_facts": [
                "logging parameters, metrics, and artifacts",
                "organizing runs into experiments",
            ]
        },
    },
    {
        "inputs": {"question": "How do I register a model in Unity Catalog?"},
        "expectations": {
            "expected_facts": [
                "mlflow.register_model with a UC three-level name",
                "set the registry URI to databricks-uc",
            ]
        },
    },
    {
        "inputs": {"question": "What does mlflow.genai.evaluate return?"},
        "expectations": {
            "expected_facts": [
                "an EvaluationResult object",
                "it contains metrics and a result DataFrame",
            ]
        },
    },
]


# ---------------------------------------------------------------------------
# 2. A single agent factory — swap models by name
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a concise, accurate assistant answering questions about MLflow "
    "and Databricks. Answer in 1-3 sentences."
)

# One OpenAI-compatible client pointed at the Databricks AI Gateway; the model
# is chosen per request, so all configs share it.
client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=AI_GATEWAY_BASE_URL,
)


def make_agent(model: str):
    """Return a traced predict_fn backed by the given Databricks-served model.

    The returned function takes the same kwargs as the dataset's `inputs`
    (here, `question`) and returns the model's answer string. ``@mlflow.trace``
    ensures every call emits exactly one trace, which the scorers evaluate.
    """

    @mlflow.trace(name=f"agent[{model}]")
    def predict_fn(question: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content

    return predict_fn


# ---------------------------------------------------------------------------
# 3. Run the sweep
# ---------------------------------------------------------------------------
# {model_name: predict_fn} — the sweep uses these keys as config labels.
# (You could also pass a plain list [make_agent(m) for m in MODELS] and the
#  sweep would name each config by the function's __name__.)
predict_fns = {model: make_agent(model) for model in MODELS}

scorers = [
    Correctness(),  # uses expectations.expected_facts
    RelevanceToQuery(),
    Safety(),
    Guidelines(
        name="conciseness",
        guidelines="The response must be at most 3 sentences.",
    ),
]

result = mlflow.genai.evaluate_sweep(
    data=EVAL_DATA,
    scorers=scorers,
    predict_fns=predict_fns,
    n_repeats=3,  # 3 runs per model -> confidence intervals
)

# ---------------------------------------------------------------------------
# 4. Inspect results
# ---------------------------------------------------------------------------
print(result)  # per-config intervals + latency summary

# Tidy quality-vs-latency table across all models and scorers.
print(result.comparison_df.to_string(index=False))

# Which model scored best on correctness?
print("Best on correctness:", result.best("correctness"))

# Drill into one config's underlying per-repeat EvaluationResult objects.
for run_result in result.configs[MODELS[0]].results:
    print(run_result.run_id, run_result.metrics)

# Everything is logged under one parent run — open it in the MLflow UI to compare
# the nested child runs side by side.
print("Parent run:", result.parent_run_id)
