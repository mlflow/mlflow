---
description: >
  Learn how to build MLflow Evaluation Datasets for your GenAI applications using various methods including from traces, from scratch, importing existing data, or synthetic generation to systematically test and improve quality.
last_update:
  date: 2025-05-18
---

# Building MLflow Evaluation Datasets

This guide shows you the various ways to create [evaluation datasets](/genai/eval-monitor/concepts/eval-datasets) in order to systematically test and improve your GenAI application's quality. You'll learn multiple approaches to build datasets that enable consistent, repeatable evaluation as you iterate on your app.

Evaluation datasets help you:

- **Fix known issues**: Add problematic examples from production to repeatedly test fixes
- **Prevent regressions**: Create a "golden set" of examples that must always work correctly
- **Compare versions**: Test different prompts, models, or app logic against the same data
- **Target specific features**: Build specialized datasets for safety, domain knowledge, or edge cases

Start with a single well-curated dataset, then expand to multiple datasets as your testing needs grow.

**What you'll learn:**

- Create datasets from production [traces](/mlflow3/genai/tracing/index) to test real-world scenarios
- Build datasets from scratch for targeted testing of specific features
- Import existing evaluation data from CSV, JSON, or other formats
- Generate synthetic test data to expand coverage
- Add ground truth labels from domain expert feedback

:::note
This guide shows you how to use MLflow-managed evaluation datasets, which provide version history and lineage tracking. For rapid prototyping, you can also provide your evaluation dataset as a Python dictionary or Pandas/Spark dataframe that follows the same schema of the MLflow-managed dataset. To learn more about the evaluation dataset schema, refer to the [evaluation datasets reference](/docs/genai/eval-monitor/concepts/eval-datasets) page.
:::

## Prerequisites

1. Install MLflow and required packages

   ```bash
   pip install --upgrade "mlflow[databricks]>=3.1.0"
   ```

2. Create an MLflow experiment by following the [setup your environment quickstart](/mlflow3/genai/getting-started/connect-environment).
3. Access to a Unity Catalog schema with `CREATE TABLE` permissions to create evaluation datasets.

   ::::aws

   :::note
   If you're using a [Databricks trial account](/getting-started/express-setup), you have CREATE TABLE permissions on the Unity Catalog schema `workspace.default`.
   :::

   ::::

## Approaches to Building Your Dataset

MLflow offers several flexible ways to construct an evaluation dataset tailored to your needs:

- **[Creating a dataset from existing traces](#creating-a-dataset-from-existing-traces):** Leverage the rich data already captured in your MLflow Traces.
- **[Importing a dataset or building a dataset from scratch](#building-a-dataset-from-scratch):** Manually define specific input examples and (optionally) expected outputs.
- **[Seeding an evaluation dataset with synthetic data](#seeding-an-evaluation-dataset-with-synthetic-data):** Generate diverse inputs automatically.

Choose the method or combination of methods that best suits your current data sources and evaluation goals.

## Step 1: Create a dataset

Irregardless of the method you choose, first, you must create a MLflow-managed evaluation dataset. This approach allows you to track changes to the dataset over time and link individual evaluation results to this dataset.

::::tabs
:::tab-item[Using the UI]

Follow the recording below to use the UI to create an evaluation dataset



:::
:::tab-item[Using the SDK]

Create an evaluation dataset programmatically by searching for traces and adding them to the dataset.

```python
import mlflow
import mlflow.genai.datasets
import time
from databricks.connect import DatabricksSession

# 0. If you are using a local development environment, connect to Serverless Spark which powers MLflow's evaluation dataset service
spark = DatabricksSession.builder.remote(serverless=True).getOrCreate()

# 1. Create an evaluation dataset

# Replace with a Unity Catalog schema where you have CREATE TABLE permission
uc_schema = "workspace.default"
# This table will be created in the above UC schema
evaluation_dataset_table_name = "email_generation_eval"

eval_dataset = mlflow.genai.datasets.create_dataset(
    uc_table_name=f"{uc_schema}.{evaluation_dataset_table_name}",
)
print(f"Created evaluation dataset: {uc_schema}.{evaluation_dataset_table_name}")
```

:::
::::

## Step 2: Add records to your dataset

### <a id="creating-a-dataset-from-existing-traces"></a>Approach 1: Create from existing traces

One of the most effective ways to build a relevant evaluation dataset is by curating examples directly from your application's historical interactions captured by MLflow Tracing. You can create datasets from traces using either the MLflow Monitoring UI or the SDK.

::::tabs
:::tab-item[Using the UI]

Follow the recording below to use the UI to add existing production traces to the dataset



:::
:::tab-item[Using the SDK]

Programmatically search for traces and then add them to the dataset. Refer to the [query traces](/mlflow3/genai/tracing/observe-with-traces/query-via-sdk#search-api) reference page for details on how to use filters in `search_traces()`.

```python
import mlflow

# 2. Search for traces
traces = mlflow.search_traces(
    filter_string="attributes.status = 'OK'",
    order_by=["attributes.timestamp_ms DESC"]
    max_results=10
)

print(f"Found {len(traces)} successful traces")

# 3. Add the traces to the evaluation dataset
eval_dataset.merge_records(traces)
print(f"Added {len(traces)} records to evaluation dataset")

# Preview the dataset
df = eval_dataset.to_df()
print(f"\nDataset preview:")
print(f"Total records: {len(df)}")
print("\nSample record:")
sample = df.iloc[0]
print(f"Inputs: {sample['inputs']}")
```

:::
::::

### <a id="creating-a-dataset-from-domain-expert-feedback"></a>Approach 2: Create from domain expert labels

Leverage feedback from domain experts captured in MLflow Labeling Sessions to enrich your evaluation datasets with ground truth labels. Before doing these steps, follow the [collect domain expert feedback](/mlflow3/genai/human-feedback/expert-feedback/label-existing-traces) guide to create a labeling session.

```python
import mlflow.genai.labeling as labeling

# Get a labeling sessions
all_sessions = labeling.get_labeling_sessions()
print(f"Found {len(all_sessions)} sessions")

for session in all_sessions:
    print(f"- {session.name} (ID: {session.labeling_session_id})")
    print(f"  Assigned users: {session.assigned_users}")

# Sync from the labeling session to the dataset

all_sessions[0].sync_expectations(dataset_name=f"{uc_schema}.{evaluation_dataset_table_name}")
```

### <a id="building-a-dataset-from-scratch"></a>Approach 3: Build from scratch or import existing

You can import an existing dataset or curate examples from scratch. Your data must match (or be transformed to match) the [evaluation dataset schema](/docs/genai/eval-monitor/concepts/eval-datasets).

```python
# Define comprehensive test cases
evaluation_examples = [
    {
        "inputs": {"question": "What is MLflow?"},
        "expected": {
            "expected_response": "MLflow is an open source platform for managing the end-to-end machine learning lifecycle.",
            "expected_facts": [
                "open source platform",
                "manages ML lifecycle",
                "experiment tracking",
                "model deployment"
            ]
        },
    },
]

eval_dataset.merge_records(evaluation_examples)
```

### <a id="seeding-an-evaluation-dataset-with-synthetic-data"></a>Approach 4: Seed using synthetic data

Generating synthetic data can expand your testing efforts by quickly creating diverse inputs and covering edge cases. To learn more, visit the [synthesize evaluation datasets](/generative-ai/agent-evaluation/synthesize-evaluation-set) reference.

## Next steps

Continue your journey with these recommended actions and tutorials.

- [Evaluate your app](/genai/eval-monitor/evaluate-app) - Use your newly created dataset for evaluation
<!-- - [Use production data for improvement](/genai/eval-monitor/continuous-improvement-with-production-data) - Create datasets from production traces -->
- [Create custom scorers](/genai/eval-monitor/custom-scorers) - Build scorers to evaluate against ground truth

## Reference guides

Explore detailed documentation for concepts and features mentioned in this guide.

- [Evaluation Datasets](/genai/eval-monitor/concepts/eval-datasets) - Deep dive into dataset structure and capabilities
- [Evaluation Harness](/genai/eval-monitor/concepts/eval-harness) - Learn how `mlflow.genai.evaluate()` uses your datasets
- [Tracing data model](/mlflow3/genai/tracing/data-model) - Understand traces as a source for evaluation datasets
