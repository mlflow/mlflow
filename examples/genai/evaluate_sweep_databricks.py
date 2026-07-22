"""Sample: sweep multiple models with ``mlflow.genai.evaluate_sweep`` on Databricks.

Run this in a Databricks notebook or against a Databricks workspace
(``mlflow.set_tracking_uri("databricks")``). It shows the intended pattern:

1. An evaluation dataset of test cases (inputs + expectations).
2. A single ``make_agent(model)`` factory that builds a traced ``predict_fn`` —
   so swapping models is just changing a string.
3. A sweep across those models with repeats, producing per-scorer confidence
   intervals and a quality-vs-latency comparison.

Set ``USE_DUMMY = True`` to run the orchestration with a deterministic fake
agent (no serving endpoint needed for the *agent*), so you can see the full flow
before pointing it at real models.

Note on scorers: the built-in scorers below (``Correctness``, ``Safety``, ...)
are LLM judges — they call a judge model, so they need model access even when
``USE_DUMMY`` is True. For a truly offline dry run, use ``OFFLINE_SCORERS``
(code-based, no LLM); flip ``USE_LLM_JUDGES`` to False for that.
"""

import os
import time

import mlflow
from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery, Safety, scorer

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
# Flip to False once you have a Databricks serving endpoint / model access.
USE_DUMMY = True
# LLM-judge scorers need a judge model even when the agent is a dummy. Set False
# for a fully offline dry run using the code-based OFFLINE_SCORERS instead.
USE_LLM_JUDGES = True

# Databricks-hosted models to compare. These are Foundation Model API endpoint
# names — swap in whatever your workspace serves. The sweep keys off these names.
MODELS = [
    "databricks-claude-sonnet-4",
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-gpt-oss-120b",
]

# On Databricks, point MLflow at the workspace. In a Databricks notebook this is
# already the default and this line is a no-op.
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


def make_agent(model: str):
    """Return a traced predict_fn backed by the given Databricks-served model.

    The returned function takes the same kwargs as the dataset's `inputs`
    (here, `question`) and returns the model's answer string. ``@mlflow.trace``
    ensures every call emits exactly one trace, which the scorers evaluate.
    """
    from openai import OpenAI

    # The Databricks workspace exposes an OpenAI-compatible endpoint. In a
    # Databricks notebook these env vars are set for you; otherwise set
    # DATABRICKS_HOST / DATABRICKS_TOKEN.
    client = OpenAI(
        api_key=os.environ["DATABRICKS_TOKEN"],
        base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
    )

    @mlflow.trace(name=f"agent[{model}]")
    def predict_fn(question: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content

    return predict_fn


def make_dummy_agent(model: str):
    """Deterministic offline stand-in, so the flow runs with no endpoint.

    Simulates model quality/latency differences so the sweep output is
    interesting: the "sonnet" model answers well, the others degrade.
    """
    canned = {
        "What is MLflow Tracking used for?": (
            "MLflow Tracking logs parameters, metrics, and artifacts, and "
            "organizes runs into experiments."
        ),
        "How do I register a model in Unity Catalog?": (
            "Set the registry URI to databricks-uc and call mlflow.register_model "
            "with a three-level UC name."
        ),
        "What does mlflow.genai.evaluate return?": (
            "It returns an EvaluationResult object containing metrics and a result DataFrame."
        ),
    }

    @mlflow.trace(name=f"dummy[{model}]")
    def predict_fn(question: str) -> str:
        # Simulate per-model latency so the latency comparison is non-trivial.
        time.sleep(0.05 if "sonnet" in model else 0.15)
        if "sonnet" in model:
            return canned.get(question, "I don't know.")
        if "llama" in model:
            # Partially correct — drops some expected facts.
            return canned.get(question, "").split(",")[0] or "Not sure."
        return "I don't know."  # weakest model

    return predict_fn


# ---------------------------------------------------------------------------
# 3. Run the sweep
# ---------------------------------------------------------------------------
factory = make_dummy_agent if USE_DUMMY else make_agent

# {model_name: predict_fn} — the sweep uses these keys as config labels.
# (You could also pass a plain list [factory(m) for m in MODELS] and the sweep
#  would name each config by the function's __name__.)
predict_fns = {model: factory(model) for model in MODELS}

# LLM-judge scorers: highest signal, but each calls a judge model.
LLM_JUDGE_SCORERS = [
    Correctness(),  # uses expectations.expected_facts
    RelevanceToQuery(),
    Safety(),
    Guidelines(
        name="conciseness",
        guidelines="The response must be at most 3 sentences.",
    ),
]


# Code-based scorers: no LLM, fully offline. A pass/fail scorer gets a Wilson
# interval in the sweep; a numeric one gets a t interval.
@scorer
def mentions_expected_fact(outputs, expectations) -> bool:
    text = str(outputs).lower()
    facts = expectations.get("expected_facts", [])
    return any(fact.lower()[:20] in text for fact in facts)


@scorer
def is_concise(outputs) -> float:
    return float(str(outputs).count(".") <= 3)


OFFLINE_SCORERS = [mentions_expected_fact, is_concise]

scorers = LLM_JUDGE_SCORERS if USE_LLM_JUDGES else OFFLINE_SCORERS

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

# Which model scored best on the first scorer that ran?
primary_scorer = scorers[0].name
print(f"Best on {primary_scorer}:", result.best(primary_scorer))

# Drill into one config's underlying per-repeat EvaluationResult objects.
for run_result in result.configs[MODELS[0]].results:
    print(run_result.run_id, run_result.metrics)

# Everything is logged under one parent run — open it in the MLflow UI to compare
# the nested child runs side by side.
print("Parent run:", result.parent_run_id)
