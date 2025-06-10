---
description: 'MLflow evaluation harness - systematically test GenAI apps with code patterns, edge cases, and advanced usage'
last_update:
  date: 2024-07-26
---

# Evaluation Harness

The `mlflow.genai.evaluate()` function systematically tests GenAI app quality by running it against test data ([evaluation datasets](/genai/eval-monitor/concepts/eval-datasets)) and applying [scorers](/genai/eval-monitor/concepts/scorers).

## Quick reference

| Parameter    | Type                                                                    | Description               |
| ------------ | ----------------------------------------------------------------------- | ------------------------- |
| `data`       | MLflow EvaluationDataset, List[Dict], Pandas DataFrame, Spark DataFrame | Test data                 |
| `predict_fn` | Callable                                                                | Your app (Mode 1 only)    |
| `scorers`    | List[Scorer]                                                            | Quality metrics           |
| `model_id`   | str                                                                     | Optional version tracking |

## How it works

1. **Runs your app** on test inputs, capturing [traces](/mlflow3/genai/tracing/index)
2. **Applies scorers** to assess quality, creating [Feedback](/mlflow3/genai/tracing/data-model#feedbacks)
3. **Stores results** in an [Evaluation Run](/genai/eval-monitor/concepts/evaluation-runs)

<!-- :::danger TODO
🔴 link api docs `mlflow.genai.evaluate`
::: -->

## Prerequisites

1. Install MLflow and required packages

   ```bash
   pip install --upgrade "mlflow[databricks]>=3.1.0" openai "databricks-connect>=16.1"
   ```

2. Create an MLflow experiment by following the [setup your environment quickstart](/mlflow3/genai/getting-started/connect-environment).

## <a id="evaluation-modes"></a>Two evaluation modes

### <a id="direct-eval"></a>Mode 1: Direct evaluation (recommended)

MLflow calls your GenAI app directly to generate and evaluate traces. You can either pass your application's entry point wrapped in a Python function ([`predict_fn`](#predict_fn)) or, if your app is deployed as a Databricks Model Serving endpoint, pass that endpoint wrapped in [`to_predict_fn`](#evaluate-a-deployed-endpoint).

**Benefits:**

- Allows scorers to be easily reused between offline evaluation and production monitoring
- Automatic parallelization of your app's execution for faster evaluation

By calling your app directly, this mode enables you to reuse the scorers defined for offline evaluation in [production monitoring](/genai/eval-monitor/concepts/production-monitoring) since the resulting traces will be identical.

<!-- ![How evaluate works with tracing](/images/mlflow3-genai/new-images/eval-with-tracing.png) -->

#### Step 1: Run evaluation

```python
import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety

# Your GenAI app with MLflow tracing
@mlflow.trace
def my_chatbot_app(question: str) -> dict:
    # Your app logic here
    if "MLflow" in question:
        response = "MLflow is an open-source platform for managing ML and GenAI workflows."
    else:
        response = "I can help you with MLflow questions."

    return {"response": response}

# Evaluate your app
results = mlflow.genai.evaluate(
    data=[
        {"inputs": {"question": "What is MLflow?"}},
        {"inputs": {"question": "How do I get started?"}}
    ],
    predict_fn=my_chatbot_app,
    scorers=[RelevanceToQuery(), Safety()]
)
```

#### Step 2: View results in the UI

![Evaluation results](https://assets.docs.databricks.com/_images/mlflow3/mlflow-screenshots/eval-overview.gif)

### <a id="answer-sheet"></a>Mode 2: Answer sheet evaluation

Provide pre-computed outputs or existing traces for evaluation when you can't run your GenAI app directly.

**Use cases:**

- Testing outputs from external systems
- Evaluating historical traces
- Comparing outputs across different platforms

:::warning
If you use an answer sheet with different traces than your production environment, you may need to re-write your scorer functions to use them for [production monitoring](/genai/eval-monitor/concepts/production-monitoring).
:::

<!-- ![How evaluate works with answer sheet](/images/mlflow3-genai/new-images/eval-with-answer-sheet.png) -->

**Example (with inputs/outputs)**:

#### Step 1: Run evaluation

```python
import mlflow
from mlflow.genai.scorers import Safety, RelevanceToQuery

# Pre-computed results from your GenAI app
results_data = [
    {
        "inputs": {"question": "What is MLflow?"},
        "outputs": {"response": "MLflow is an open-source platform for managing machine learning workflows, including tracking experiments, packaging code, and deploying models."},
    },
    {
        "inputs": {"question": "How do I get started?"},
        "outputs": {"response": "To get started with MLflow, install it using 'pip install mlflow' and then run 'mlflow ui' to launch the web interface."},
    }
]

# Evaluate pre-computed outputs
evaluation = mlflow.genai.evaluate(
    data=results_data,
    scorers=[Safety(), RelevanceToQuery()]
)
```

#### Step 2: View results in the UI

![Evaluation results](https://assets.docs.databricks.com/_images/mlflow3/mlflow-screenshots/eval-overview.gif)

**Example with existing traces:**

```python
import mlflow

# Retrieve traces from production
traces = mlflow.search_traces(
    filter_string="trace.status = 'OK'",
)

# Evaluate problematic traces
evaluation = mlflow.genai.evaluate(
    data=traces,
    scorers=[Safety(), RelevanceToQuery()]
)
```

## <a id="key-parameters"></a>Key parameters

```python
def mlflow.genai.evaluate(
    data: Union[pd.DataFrame, List[Dict], mlflow.genai.datasets.EvaluationDataset],
    scorers: list[mlflow.genai.scorers.Scorer],
    predict_fn: Optional[Callable[..., Any]] = None,
    model_id: Optional[str] = None,
) -> mlflow.models.evaluation.base.EvaluationResult:
```

### <a id="data-parameter"></a>`data`

Your evaluation dataset in one of these formats:

- `EvaluationDataset` (recommended)
- List of dictionaries, Pandas DataFrame, or Spark DataFrame

If the data argument is provided as a DataFrame or list of dictionaries, it must follow the following schema. This is consistent with the schema used by [EvaluationDataset](/genai/eval-monitor/build-eval-dataset). We recommend using an `EvaluationDataset` as it will enforce schema validation, in addition to tracking the lineage of each record.

| Field          | Data type               | Description                                                                                                                                                                                                                 | Required if [app is passed to `predict_fn` (mode 1)](#mode-1-direct-evaluation-recommended)? | Required if [providing an answer sheet (mode 2)](#mode-2-answer-sheet-evaluation)?                                    |
| -------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `inputs`       | `dict[Any, Any]`        | A `dict` that will be passed to your `predict_fn` using `**kwargs`. Must be JSON serializable. Each key must correspond to a named argument in `predict_fn`.                                                                | Required                                                                                     | Either `inputs` + `outputs` or `trace` is required. Cannot pass both. <br/><br/>Derived from `trace` if not provided. |
| `outputs`      | `dict[Any, Any]`        | A `dict` with the outputs of your GenAI app for the corresponding `input`. Must be JSON serializable.                                                                                                                       | Must NOT be provided, generated by MLflow from the Trace                                     | Either `inputs` + `outputs` or `trace` is required. Cannot pass both. <br/><br/>Derived from `trace` if not provided. |
| `expectations` | `dict[str, Any]`        | A `dict` with ground-truth labels corresponding to `input`. Used by `scorers` to check quality. Must be JSON serializable and each key must be a `str`.                                                                     | Optional                                                                                     | Optional                                                                                                              |
| `trace`        | `mlflow.entities.Trace` | The trace object for the request. If the `trace` is provided, the `expectations` can be provided as [`Assessments`](/mlflow3/genai/tracing/collect-user-feedback/index) on the `trace` rather than as a separate column. | Must NOT be provided, generated by MLflow from the Trace                                     | Either `inputs` + `outputs` or `trace` is required. Cannot pass both.                                                 |

### `predict_fn`

Your GenAI app's entry point ([Mode 1 only](#mode-1-direct-evaluation-recommended)). Must:

- Accept the keys from the `inputs` dictionary in `data` as keyword arguments
- Return a JSON-serializable dictionary
- Be instrumented with [MLflow Tracing](/mlflow3/genai/tracing/index)
- Emit exactly one trace per call

### <a id="scorer-parameter"></a>`scorers`

List of quality metrics to apply. You can provide:

- [Predefined scorers](/genai/eval-monitor/concepts/judges/pre-built-judges-scorers)
- [Custom scorers](/genai/eval-monitor/custom-scorers)

See [Scorers](/genai/eval-monitor/concepts/scorers) for more details.

### `model_id`

Optional model identifier to link results to your app version (e.g., `"models:/my-app/1"`). See [Version Tracking](/mlflow3/genai/prompt-version-mgmt/version-tracking/track-application-versions-with-mlflow) for more details.

## <a id="data-formats"></a>Data formats

### For direct evaluation (Mode 1)

| Field          | Required | Description                            |
| -------------- | -------- | -------------------------------------- |
| `inputs`       | ✅       | Dictionary passed to your `predict_fn` |
| `expectations` | Optional | Optional ground truth for scorers      |

### For answer sheet evaluation (Mode 2)

**Option A - Provide inputs and outputs:**

| Field          | Required | Description                        |
| -------------- | -------- | ---------------------------------- |
| `inputs`       | ✅       | Original inputs to your GenAI app  |
| `outputs`      | ✅       | Pre-computed outputs from your app |
| `expectations` | Optional | Optional ground truth for scorers  |

**Option B - Provide existing traces:**

| Field          | Required | Description                              |
| -------------- | -------- | ---------------------------------------- |
| `trace`        | ✅       | MLflow Trace objects with inputs/outputs |
| `expectations` | Optional | Optional ground truth for scorers        |

## <a id="common-patterns-data"></a>Common data input patterns

### <a id="eval-dataset"></a>Evaluate with a MLflow Evaluation Dataset (recommended)

MLflow Evaluation Datasets provide versioning, lineage tracking, and Unity Catalog integration for production-ready evaluation.

```python
import mlflow
from mlflow.genai.scorers import Correctness, Safety
from my_app import agent  # Your GenAI app with tracing

# Load versioned evaluation dataset
dataset = mlflow.genai.datasets.get_dataset("catalog.schema.eval_dataset_name")

# Run evaluation
results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=agent,
    scorers=[Correctness(), Safety()],
)
```

**Use for:**

- Need to have evaluation data with version control and lineage tracking
- Easily converting traces to evaluation records

See [Build evaluation datasets](/genai/eval-monitor/build-eval-dataset) to create datasets from traces or scratch.

### <a id="eval-list-dict"></a>Evaluate with a list of dictionaries

Use a simple list of dictionaries for quick prototyping without creating a formal evaluation dataset.

```python
import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery
from my_app import agent  # Your GenAI app with tracing

# Define test data as a list of dictionaries
eval_data = [
    {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {"expected_facts": ["open-source platform", "ML lifecycle management"]}
    },
    {
        "inputs": {"question": "How do I track experiments?"},
        "expectations": {"expected_facts": ["mlflow.start_run()", "log metrics", "log parameters"]}
    },
    {
        "inputs": {"question": "What are MLflow's main components?"},
        "expectations": {"expected_facts": ["Tracking", "Projects", "Models", "Registry"]}
    }
]

# Run evaluation
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=agent,
    scorers=[Correctness(), RelevanceToQuery()],
)
```

**Use for:**

- Quick prototyping
- Small datasets (< 100 examples)
- Ad-hoc development testing

For production, convert to an [MLflow Evaluation Dataset](/genai/eval-monitor/build-eval-dataset).

### <a id="eval-pandas-dataframe"></a>Evaluate with a Pandas DataFrame

Use Pandas DataFrames for evaluation when working with CSV files or existing data science workflows.

```python
import mlflow
import pandas as pd
from mlflow.genai.scorers import Correctness, Safety
from my_app import agent  # Your GenAI app with tracing

# Create evaluation data as a Pandas DataFrame
eval_df = pd.DataFrame([
    {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {"expected_response": "MLflow is an open-source platform for ML lifecycle management"}
    },
    {
        "inputs": {"question": "How do I log metrics?"},
        "expectations": {"expected_response": "Use mlflow.log_metric() to log metrics"}
    }
])

# Run evaluation
results = mlflow.genai.evaluate(
    data=eval_df,
    predict_fn=agent,
    scorers=[Correctness(), Safety()],
)
```

**Use for:**

- Quick prototyping
- Small datasets (< 100 examples)
- Ad-hoc development testing

### <a id="eval-spark-dataframe"></a>Evaluate with a Spark DataFrame

Use Spark DataFrames for large-scale evaluations or when data is already in Delta Lake/Unity Catalog.

```python
import mlflow
from mlflow.genai.scorers import Safety, RelevanceToQuery
from my_app import agent  # Your GenAI app with tracing

# Load evaluation data from a Delta table in Unity Catalog
eval_df = spark.table("catalog.schema.evaluation_data")

# Or load from any Spark-compatible source
# eval_df = spark.read.parquet("path/to/evaluation/data")

# Run evaluation
results = mlflow.genai.evaluate(
    data=eval_df,
    predict_fn=agent,
    scorers=[Safety(), RelevanceToQuery()],
)
```

**Use for:**

- Data exists already in Delta Lake or Unity Catalog
- If you need to filter the records in an MLflow Evaluation Dataset before running evaluation

**Note:** DataFrame must comply with the [evaluation dataset schema](/genai/eval-monitor/concepts/eval-datasets).

## <a id="common-patterns-predict-fn"></a>Common `predict_fn` patterns

### <a id="eval-app-direct"></a>Call your app directly

Pass your app directly as `predict_fn` when parameter names match your evaluation dataset keys.

```python
import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety

# Your GenAI app that accepts 'question' as a parameter
@mlflow.trace
def my_chatbot_app(question: str) -> dict:
    # Your app logic here
    response = f"I can help you with: {question}"
    return {"response": response}

# Evaluation data with 'question' key matching the function parameter
eval_data = [
    {"inputs": {"question": "What is MLflow?"}},
    {"inputs": {"question": "How do I track experiments?"}}
]

# Pass your app directly since parameter names match
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=my_chatbot_app,  # Direct reference, no wrapper needed
    scorers=[RelevanceToQuery(), Safety()]
)
```

**Use for:**

- Apps that have parameter names that match your evaluation dataset's `inputs`

### <a id="eval-app-wrapper"></a>Wrap your app in a callable

Wrap your app when it expects different parameter names or data structures than your evaluation dataset's `inputs`.

```python
import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety

# Your existing GenAI app with different parameter names
@mlflow.trace
def customer_support_bot(user_message: str, chat_history: list = None) -> dict:
    # Your app logic here
    context = f"History: {chat_history}" if chat_history else "New conversation"
    return {
        "bot_response": f"Helping with: {user_message}. {context}",
        "confidence": 0.95
    }

# Wrapper function to translate evaluation data to your app's interface
def evaluate_support_bot(question: str, history: str = None) -> dict:
    # Convert evaluation dataset format to your app's expected format
    chat_history = history.split("|") if history else []

    # Call your app with the translated parameters
    result = customer_support_bot(
        user_message=question,
        chat_history=chat_history
    )

    # Translate output to standard format if needed
    return {
        "response": result["bot_response"],
        "confidence_score": result["confidence"]
    }

# Evaluation data with different key names
eval_data = [
    {"inputs": {"question": "Reset password", "history": "logged in|forgot email"}},
    {"inputs": {"question": "Track my order"}}
]

# Use the wrapper function for evaluation
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=evaluate_support_bot,  # Wrapper handles translation
    scorers=[RelevanceToQuery(), Safety()]
)
```

**Use for:**

- Parameter name mismatches between your app's parameters and evaluation dataset `input` keys (e.g., `user_input` vs `question`)
- Data format conversions (string to list, JSON parsing)

### <a id="eval-endpoint"></a>Evaluate a deployed endpoint

For Databricks Agent Framework or Model Serving endpoints, use `to_predict_fn` to create a compatible predict function.

```python
import mlflow
from mlflow.genai.scorers import RelevanceToQuery

# Create predict function for your endpoint
predict_fn = mlflow.genai.to_predict_fn("endpoints:/my-chatbot-endpoint")

# Evaluate
results = mlflow.genai.evaluate(
    data=[{"inputs": {"question": "How does MLflow work?"}}],
    predict_fn=predict_fn,
    scorers=[RelevanceToQuery()]
)
```

**Benefit:** Automatically extracts traces from tracing-enabled endpoints for full observability.

### <a id="eval-model"></a>Evaluate a logged model

Wrap logged MLflow models to translate between evaluation's named parameters and the model's single-parameter interface.

Most logged models (such as those using PyFunc or logging flavors like LangChain) accept a single input parameter (e.g., `model_inputs` for PyFunc), while `predict_fn` expects named parameters that correspond to the keys in your evaluation dataset.

```python
import mlflow
from mlflow.genai.scorers import Safety

# Make sure to load your logged model outside of the predict_fn so MLflow only loads it once!
model = mlflow.pyfunc.load_model("models:/chatbot/staging")

def evaluate_model(question: str) -> dict:
    return model.predict({"question": question})

results = mlflow.genai.evaluate(
    data=[{"inputs": {"question": "Tell me about MLflow"}}],
    predict_fn=evaluate_model,
    scorers=[Safety()]
)
```

## Next Steps

- [Evaluate your app](/genai/eval-monitor/evaluate-app) - Step-by-step guide to running your first evaluation
- [Build evaluation datasets](/genai/eval-monitor/build-eval-dataset) - Create structured test data from production logs or scratch
- [Define custom scorers](/genai/eval-monitor/custom-scorers) - Build metrics tailored to your specific use case
