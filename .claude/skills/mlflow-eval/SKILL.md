---
name: mlflow-eval
description: Add MLflow tracing and evaluation to a GenAI agent or LLM app. Use when the user is building, debugging, or evaluating an LLM-powered application and wants observability, scoring, or judge-based evaluation.
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# MLflow GenAI Evaluation

Use MLflow to instrument LLM apps and agents with tracing, then score them with built-in or custom judges. Use this skill when the user is wiring observability or evaluation into a GenAI app.

## When to use

Use this skill when the user:

- Asks "how do I add tracing / eval / judges to my agent"
- Wants to measure agent quality on a dataset
- Has a working agent and wants to optimize prompts against an objective metric
- Asks about subjective criteria (formality, conciseness, helpfulness) and how to make a judge agree with humans

Skip this skill when the user is working on MLflow internals (the existing `pr-review`, `resolve`, `copilot` skills cover that).

## The flow

### 1. Trace the app

Wrap the entry point so every LLM call, tool call, and step is recorded.

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-agent")


@mlflow.trace
def run_agent(user_input: str) -> str:
    ...
```

For LangChain, OpenAI, Anthropic, LiteLLM, and similar SDKs, prefer autolog instead of hand-tracing each call:

```python
mlflow.openai.autolog()
mlflow.anthropic.autolog()
mlflow.langchain.autolog()
```

Each run produces a structured trace with the full span tree (tool calls, retrieval, LLM I/O), visible in the MLflow UI.

### 2. Score with built-in scorers

For common criteria, MLflow ships built-in `mlflow.genai.scorers`. Use a built-in when the criterion matches; otherwise build a custom judge in step 3.

| Scorer | Measures | Level |
| --- | --- | --- |
| `Correctness` | Output matches expected | turn |
| `Safety` | No toxic / unsafe content | turn |
| `RelevanceToQuery` | Output addresses the query | turn |
| `Guidelines` | Output follows free-text guidelines | turn |
| `RetrievalGroundedness` | Output is grounded in retrieved context | turn |
| `RetrievalRelevance` | Retrieved context is relevant | turn |
| `RetrievalSufficiency` | Retrieved context is enough to answer | turn |

Session-level multi-turn scorers (`ConversationalCoherence`, `AgentPlanQuality`) are tracked in [#22626](https://github.com/mlflow/mlflow/pull/22626); use them once that PR lands.

### 3. Build a custom judge with `make_judge`

For domain-specific or subjective criteria, write a prompt-based judge:

```python
from mlflow.genai import make_judge

formality_judge = make_judge(
    name="formality",
    instructions=(
        "Score whether the response in {{ outputs }} maintains a formal, "
        "professional tone given the request in {{ inputs }}. "
        "Return true if it does, false otherwise, with a short rationale."
    ),
    feedback_value_type=bool,
    model="openai:/gpt-4o-mini",
)
```

Allowed template variables: `{{ inputs }}`, `{{ outputs }}`, `{{ expectations }}`, `{{ conversation }}`, `{{ trace }}`. `{{ conversation }}` cannot be combined with `{{ inputs }}`, `{{ outputs }}`, or `{{ trace }}`.

`feedback_value_type` controls the judge's output type: `bool` for pass/fail, `int` for ordinal scales (1-5), `float` for continuous scores, `Literal["a","b","c"]` for enum, or `str` for free text.

### 4. Align the judge to human preferences (turn-level only)

Subjective criteria (formality, conciseness, helpfulness) often disagree with the LLM's prior. Align the judge against feedback collected on real traces:

```python
# After collecting human feedback on a set of traces:
aligned = formality_judge.align(traces=feedback_traces)
```

Use `aligned` in place of `formality_judge` in your evaluation.

By default this uses `SIMBAAlignmentOptimizer`; pass `optimizer=` to swap. The available alignment optimizers exported from `mlflow.genai.judges.optimizers` are `SIMBAAlignmentOptimizer`, `GEPAAlignmentOptimizer`, and `MemAlignOptimizer`.

### 5. Run evaluation

`mlflow.genai.evaluate` calls `predict_fn` with each row's `inputs` **as keyword arguments**, so the dict keys must match the `predict_fn` parameter names exactly. For `run_agent(user_input: str)` the dataset looks like:

```python
import pandas as pd
import mlflow.genai
from mlflow.genai.scorers import Correctness, Safety

eval_dataset = pd.DataFrame(
    [
        {
            "inputs": {"user_input": "What is MLflow?"},
            "expectations": "MLflow is an open-source ML lifecycle platform.",
        },
        {
            "inputs": {"user_input": "What is Spark?"},
            "expectations": "Spark is a distributed data processing engine.",
        },
    ]
)

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=run_agent,
    scorers=[Correctness(), Safety(), aligned],
)
```

Accepted `data` shapes: a pandas/Spark DataFrame, a list of dicts, an `EvaluationDataset`, or a DataFrame of existing traces from `mlflow.search_traces(...)` (in which case `predict_fn` is omitted and scorers read from the trace). Results land in the active experiment under Evaluations and are diff-able across agent versions.

### 6. Optimize the agent's prompt

Once you have a trustworthy objective signal, refine the prompt automatically:

```python
import mlflow
from mlflow.genai.optimize import optimize_prompts
from mlflow.genai.optimize.optimizers import GepaPromptOptimizer
from mlflow.genai.scorers import Correctness

prompt = mlflow.genai.register_prompt(
    name="research-agent",
    template="Answer the question: {{ question }}",
)

result = optimize_prompts(
    predict_fn=run_agent,
    train_data=train_dataset,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-4o"),
    scorers=[Correctness()],
)
```

The `predict_fn` must call `PromptVersion.format` on a prompt referenced in `prompt_uris` for the optimizer to know which prompt text to rewrite. `optimize_prompts` rewrites prompt text only — to rewrite tools or rewire agent logic, drive the loop from Claude Code with this skill instead.

## Decision tree

- New LLM app, no observability yet → step 1 (trace).
- Have traces, need a quality signal → step 2 (built-in) or step 3 (`make_judge`).
- Subjective criterion the LLM judges wrong → step 4 (align), turn-level only.
- Have an objective signal, want a better agent → step 6 (optimize).

## Gotchas

- Call `mlflow.set_tracking_uri()` before any `@mlflow.trace`-decorated function runs, otherwise spans go to the default local store.
- `judge.align(...)` raises `NotImplementedError` on session-level scorers — alignment is currently turn-level only.
- Plan for at least ~30 feedback samples before alignment produces a meaningful refined judge.
- `make_judge` instructions must contain at least one template variable, otherwise the call fails validation.
- `make_judge(model=...)` accepts URIs like `"openai:/gpt-4o-mini"` or `"databricks:/..."`. `base_url` and `extra_headers` are not supported for Databricks-backed models.
- For reproducible eval runs, pin the judge's sampling: `make_judge(..., inference_params={"temperature": 0})`. Without this, judge scores drift between runs and break diff-based regression checks.
- `mlflow.genai.evaluate(predict_fn=...)` invokes `predict_fn` with `**inputs`, so each row's `inputs` dict keys must match the function's parameter names. A signature mismatch surfaces as `TypeError: got an unexpected keyword argument`, not a clear validation error.
- Session-level scorers group traces that share a `session_id` (set on the trace's metadata, or as a `session_id` column on the evaluation dataset). A single trace — however many spans it contains — is still a turn-level trace, not a session.
- `optimize_prompts` only edits prompt text. For tool / agent-logic rewrites, use Claude Code with this skill loaded.

## See also

- [Custom judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges/create-custom-judge/)
- [Detect issues automatically](https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/detect-issues)
- [Align judges with feedback](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/align-judges)
- [Logging assessments / feedback](https://mlflow.org/docs/latest/genai/assessments/feedback/)
- [coSTAR pattern](https://github.com/alkispoly-db/costar) — end-to-end Scenario→Trace→Assess→Refine loops with MLflow + Claude Code
