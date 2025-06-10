---
description: >
  Learn how to set up automated quality monitoring for your GenAI applications in MLflow by scheduling scorers to run on production traces, enabling continuous assessment of application quality.
last_update:
  date: 2025-05-18
---

# Production quality monitoring (running scorers automatically)

::include[beta]

MLflow enables you to automatically run scorers on a sample of your production traces to continuously monitor quality.

Key benefits:

- **Automated quality assessment** without manual intervention
- **Flexible sampling** to balance coverage with computational cost
- **Consistent evaluation** using the same scorers from development
- **Continuous monitoring** with periodic background execution

## Prerequisites

1. Install MLflow and required packages

   ```bash
   pip install --upgrade "mlflow[databricks]>=3.1.0" openai
   ```

2. Create an MLflow experiment by following the [setup your environment quickstart](/mlflow3/genai/getting-started/connect-environment).
3. [Instrumented](/mlflow3/genai/tracing/prod-tracing) your production application with MLflow tracing
4. Access to a Unity Catalog schema with `CREATE TABLE` permissions in order to store the monitoring outputs.

   ::::aws

   :::note
   If you are using a [Databricks trial account](/getting-started/express-setup), you have CREATE TABLE permissions on the Unity Catalog schema `workspace.default`.
   :::

   ::::

## Step 1: Test scorers on your production traces

First, we need to test that the scorers you will use in production can evaluate your traces.

:::tip
If you used your production app as the `predict_fn` in `mlflow.genai.evaluate()` during development, your scorers are likely already compatible.
:::

:::warning
MLflow currently only supports using [predefined scorers](/genai/eval-monitor/predefined-judge-scorers) for production monitoring. Contact your Databricks account represenative if you need to run custom code-based or LLM-based scorers in production.
:::

1. Use `mlflow.genai.evaluate()` to test the scorers on a sample of your traces

   ```python
   import mlflow

   from mlflow.genai.scorers import (
       Guidelines,
       RelevanceToQuery,
       RetrievalGroundedness,
       RetrievalRelevance,
       Safety,
   )

   # Get a sample of up to 10 traces from your experiment
   traces = mlflow.search_traces(max_results=10)

   # Run evaluation to test the scorers
   mlflow.genai.evaluate(
       data=traces,
       scorers=[
           RelevanceToQuery(),
           RetrievalGroundedness(),
           RetrievalRelevance(),
           Safety(),
           Guidelines(
               name="mlflow_only",
               # Guidelines can refer to the request and response.
               guidelines="If the request is unrelated to MLflow, the response must refuse to answer.",
           ),
           # You can have any number of guidelines.
           Guidelines(
               name="customer_service_tone",
               guidelines="""The response must maintain our brand voice which is:
       - Professional yet warm and conversational (avoid corporate jargon)
       - Empathetic, acknowledging emotional context before jumping to solutions
       - Proactive in offering help without being pushy

       Specifically:
       - If the customer expresses frustration, anger, or disappointment, the first sentence must acknowledge their emotion
       - The response must use "I" statements to take ownership (e.g., "I understand" not "We understand")
       - The response must avoid phrases that minimize concerns like "simply", "just", or "obviously"
       - The response must end with a specific next step or open-ended offer to help, not generic closings""",
           ),
       ],
   )
   ```

2. Use the MLflow Trace UI to check which scorers ran

   In this case, we notice that even though we ran the `RetrievalGroundedness()` and `RetrievalRelevance()` scorers, they did not show up in the MLflow UI. This indicates these scorers do not work with our traces and thus, we should not enable them in the next step.

## Step 2: Enable monitoring

Now, let's enable the monitoring service. Once enabled, the monitoring service will sync a copy of your evaluated traces from your MLflow Experiment to a Delta Table in the Unity Catalog schema you specify.

:::important
Once set, the Unity Catalog schema can not be changed.
:::

::::tabs
:::tab-item[Using the UI]

Follow the recording below to use the UI to enable the scorers that successfully ran in step 1. Selecting a sampling rate only runs the scorers on that percentage of traces (e.g., entering `1.0` will run the scorers on 100% of your traces and `.2` will run on 20%, etc).

If you want to set the sampling rate per-scorer, you must use the SDK.

:::
:::tab-item[Using the SDK]

Use the below code snippet to enable the scorers that successfully ran in step 1. Selecting a sampling rate only runs the scorers on that percentage of traces (e.g., entering `1.0` will run the scorers on 100% of your traces and `.2` will run on 20%, etc). Optionally, you can configure the sampling rate per scorer.

```python
# These packages are automatically installed with mlflow[databricks]
from databricks.agents.monitoring import create_external_monitor, AssessmentsSuiteConfig, BuiltinJudge, GuidelinesJudge

external_monitor = create_external_monitor(
    # Change to a Unity Catalog schema where you have CREATE TABLE permissions.
    catalog_name="workspace",
    schema_name="default",
    assessments_config=AssessmentsSuiteConfig(
        sample=1.0,  # sampling rate
        assessments=[
            # Predefined scorers "safety", "groundedness", "relevance_to_query", "chunk_relevance"
            BuiltinJudge(name="safety"),  # or {'name': 'safety'}
            BuiltinJudge(
                name="groundedness", sample_rate=0.4
            ),  # or {'name': 'groundedness', 'sample_rate': 0.4}
            BuiltinJudge(
                name="relevance_to_query"
            ),  # or {'name': 'relevance_to_query'}
            BuiltinJudge(name="chunk_relevance"),  # or {'name': 'chunk_relevance'}
            # Guidelines can refer to the request and response.
            GuidelinesJudge(
                guidelines={
                    # You can have any number of guidelines, each defined as a key-value pair.
                    "mlflow_only": [
                        "If the request is unrelated to MLflow, the response must refuse to answer."
                    ],  # Must be an array of strings
                    "customer_service_tone": [
                        """The response must maintain our brand voice which is:
    - Professional yet warm and conversational (avoid corporate jargon)
    - Empathetic, acknowledging emotional context before jumping to solutions
    - Proactive in offering help without being pushy

    Specifically:
    - If the customer expresses frustration, anger, or disappointment, the first sentence must acknowledge their emotion
    - The response must use "I" statements to take ownership (e.g., "I understand" not "We understand")
    - The response must avoid phrases that minimize concerns like "simply", "just", or "obviously"
    - The response must end with a specific next step or open-ended offer to help, not generic closings"""
                    ],
                }
            ),
        ],
    ),
)

print(external_monitor)
```

:::
::::

## Step 3. Updating your monitor

To change the scorers configuration, use `update_external_monitor()`. The configuration is stateless - that is, it is completely overwritten by the update. To retrieve an existing configuration to modify, use `get_external_monitor()`.

::::tabs
:::tab-item[Using the UI]

Follow the recording below to use the UI to update the scorers.

:::
:::tab-item[Using the SDK]

```python
# These packages are automatically installed with mlflow[databricks]
from databricks.agents.monitoring import update_external_monitor, get_external_monitor
import os

config = get_external_monitor(experiment_id=os.environ["MLFLOW_EXPERIMENT_ID"])
print(config)


external_monitor = update_external_monitor(
    # You must pass the experiment_id of the experiment you want to update.
    experiment_id=os.environ["MLFLOW_EXPERIMENT_ID"],
    # Change to a Unity Catalog schema where you have CREATE TABLE permissions.
    assessments_config=AssessmentsSuiteConfig(
        sample=1.0,  # sampling rate
        assessments=[
            # Predefined scorers "safety", "groundedness", "relevance_to_query", "chunk_relevance"
            BuiltinJudge(name="safety"),  # or {'name': 'safety'}
            BuiltinJudge(
                name="groundedness", sample_rate=0.4
            ),  # or {'name': 'groundedness', 'sample_rate': 0.4}
            BuiltinJudge(
                name="relevance_to_query"
            ),  # or {'name': 'relevance_to_query'}
            BuiltinJudge(name="chunk_relevance"),  # or {'name': 'chunk_relevance'}
            # Guidelines can refer to the request and response.
            GuidelinesJudge(
                guidelines={
                    # You can have any number of guidelines, each defined as a key-value pair.
                    "mlflow_only": [
                        "If the request is unrelated to MLflow, the response must refuse to answer."
                    ],  # Must be an array of strings
                    "customer_service_tone": [
                        """The response must maintain our brand voice which is:
    - Professional yet warm and conversational (avoid corporate jargon)
    - Empathetic, acknowledging emotional context before jumping to solutions
    - Proactive in offering help without being pushy

    Specifically:
    - If the customer expresses frustration, anger, or disappointment, the first sentence must acknowledge their emotion
    - The response must use "I" statements to take ownership (e.g., "I understand" not "We understand")
    - The response must avoid phrases that minimize concerns like "simply", "just", or "obviously"
    - The response must end with a specific next step or open-ended offer to help, not generic closings"""
                    ],
                }
            ),
        ],
    ),
)

print(external_monitor)
```

:::
::::

## Step 4. Use monitoring results

The monitoring job will take ~15 - 30 minutes to run for the first time. After the initial run, it runs every 15 minutes. Note that if you have a large volume of production traffic, the job can take additional time to complete.

Each time the job runs, it:

1. Runs each scorer on the sample of traces
   - If you have different sampling rates per scorer, the monitoring job attempts to score as many of the same traces as possible. For example, if scorer A has a 20% sampling rate and scorer B has a 40% sampling rate, the same 20% of traces will be used for A and B.
2. Attaches the [feedback](/mlflow3/genai/tracing/data-model#feedback) from the scorer to each trace in the MLflow Experiment
3. Writes a copy of ALL traces (not just the ones sampled) to the Delta Table configured in Step 1.

You can view the monitoring results using the Trace tab in the MLflow Experiment:

Alternatively, you can query the traces using SQL or Spark in the generated Delta Table.

## Next steps

Continue your journey with these recommended actions and tutorials.

- [Use production traces to improve your app's quality](/genai/eval-monitor/evaluate-app) - Create semantic evaluation using LLMs
- [Build evaluation datasets](/genai/eval-monitor/build-eval-dataset) - Use the results of monitoring to curate low performing traces into evaluation datasets to improve their quality.

## Reference guides

Explore detailed documentation for concepts and features mentioned in this guide.

- [Production Monitoring](/genai/eval-monitor/concepts/production-monitoring) - Deep dive into the production monitoring SDKs
