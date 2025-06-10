---
description: >
  Learn how to use production data and MLflow Tracing for continuous improvement of your GenAI application, from monitoring in production to evaluating and fixing issues in development.
last_update:
  date: 2025-05-18
---

# Continuous Improvement of GenAI Apps with Production Data

<!--
:::danger UNDER CONSTRUCTION
🔴 This page is under construction and content may be incomplete or subject to change.
:::

Continuously improving your GenAI application involves a cycle of monitoring its performance in production, identifying areas for enhancement, and then using that real-world data to guide development and testing. Production traffic and traces are invaluable resources, offering direct insights into how your application performs, where it excels, and where it falls short. By systematically analyzing this data and integrating it into your development workflow, you can create a powerful feedback loop to debug issues, refine prompts, enhance retrieval strategies, and ultimately deliver a better user experience.

MLflow, with its comprehensive tracing and evaluation capabilities, supports this entire continuous improvement lifecycle.

## Why Use Production Data for Continuous Improvement?

- **Real-World Relevance:** Production data reflects actual user behavior, ensuring your improvement efforts are focused on real issues and opportunities.
- **Identify Blind Spots:** Uncover edge cases, unexpected query patterns, or areas of low quality that weren't anticipated during development.
- **Data for Iteration & Fine-Tuning:** Curate examples from production (both good and bad) to create datasets for offline evaluation, regression testing, and potentially fine-tuning your LLMs or retrieval models.
- **Debug Complex Issues:** Individual traces provide detailed context for diagnosing specific failures or poor responses reported by users or detected through monitoring.
- **Understand User Intent:** Analyze popular queries and interaction patterns to better understand what users are trying to achieve and how your application can better meet their needs.
- **Maintain Quality Over Time:** LLM behavior can drift, data distributions can change, and new user interaction patterns can emerge. Continuous monitoring and feedback help detect and address quality degradation.
- **Ensure Operational Health:** Track performance metrics like latency, error rates, and costs to ensure your application is running efficiently and reliably.

## The Continuous Improvement Workflow

The core workflow involves monitoring your GenAI application in production, using those insights to identify issues or areas for improvement, and then bringing relevant production data back into your development environment for analysis, evaluation, and refinement.

### 1. Monitor Your Application in Production

Once your GenAI application is deployed, continuous production monitoring is essential. It extends evaluation practices from development into the live environment by tracking key operational metrics, automatically assessing quality on production traffic, and visualizing these insights.

#### Key Aspects of Production Monitoring:

- **Operational Metrics (Latency, Errors, Cost, etc.):**

  - Track essential operational health indicators like request latency (end-to-end, LLM, retrieval), error rates (application, LLM, dependency), throughput, and costs (token consumption, API costs).
  - MLflow Tracing automatically captures timestamps (for latency) and status (success/failure). You can log custom operational data like token counts or error details using `mlflow.set_tags()` or by logging metrics.
  - Analyzing these metrics helps maintain user experience, system stability, manage costs, and guide performance optimization.

- **Quality Monitoring (Running Scorers Automatically):**

  - Schedule predefined and custom quality scorers (LLM judges, heuristic-based metrics) to run automatically on a sample of production traces. This provides ongoing assessment of your application's quality dimensions (e.g., relevance, groundedness, safety, guideline adherence).
  - Refer to [Define Quality Metrics and Scorers](/genai/eval-monitor/concepts/scorers) and [Automated Quality Scoring in Production](/genai/eval-monitor/run-scorer-in-prod) for details.

- **Custom Dashboards and Alerting:**
  - Leverage the data collected from operational and quality monitoring to build custom dashboards (e.g., using Databricks SQL Dashboards, Grafana). This offers tailored views for different stakeholders and facilitates trend analysis. See [Creating Custom Dashboards](/mlflow3/genai/tracing/build-custom-dashboards).
  - Set up alerts based on metric thresholds (e.g., P95 latency > X seconds, relevance score < Y) to be notified of problems quickly, enabling timely interventions.

### 2. Identify and Select Relevant Traces for Improvement

When monitoring reveals issues (e.g., a dip in a quality score, a spike in errors, negative user feedback, or high-latency operations), the next step is to investigate.

- **Source:** Access your production (or staging/pre-production) trace logs captured by MLflow Tracing.
- **Selection Criteria:** Identify traces that are particularly interesting for deeper analysis and potential use in development. This might include:
  - Traces with negative user feedback (e.g., logged via `mlflow.log_feedback()`).
  - Traces where automated quality judges indicated poor performance.
  - Traces that resulted in errors or unexpected application behavior.
  - Frequently occurring query patterns, especially those leading to suboptimal responses.
  - Examples of successful interactions that you want to ensure don't regress (for regression testing).
  - Traces representing critical user journeys or business functionalities.
- **Tools for Selection:**
  - Use the MLflow UI (Trace View or Logs tab in Monitoring, if available) to visually inspect and filter traces.
  - Programmatically query traces using `mlflow.search_traces()` in the SDK. You can filter based on inputs, outputs, metadata (tags), assessments, or timestamps.
  ```python
  # Example: Find traces with negative feedback for a specific feature
  # import mlflow
  # problematic_traces_df = mlflow.search_traces(
  #     filter_string="inputs.query CONTAINS 'Feature X' AND assessments.user_feedback.value = 'negative'"
  # )
  ```

### 3. Bring Production Insights into Development

Once relevant traces are identified, bring them into your development environment to guide improvements. This often involves creating or augmenting evaluation datasets.

- **Direct Ad-hoc Evaluation:** For quick checks, you can take a small sample of production inputs (requests) and run them directly against a development version of your application or a new prompt you're testing. This is less formal but provides rapid feedback.

- **Create/Augment Formal Evaluation Datasets:** For systematic and repeatable evaluation, the best practice is to convert selected production traces into an MLflow Evaluation Dataset. This involves:
  - Extracting relevant fields (inputs, outputs from production, request IDs, any existing human labels or production judge scores).
  - Structuring this data according to the `mlflow.data.EvaluationDataset` schema.
  - Logging this dataset to MLflow.
  - This allows you to systematically test new versions of your app with `mlflow.evaluate()`.
  - Refer to [Creating a dataset from existing traces](/genai/eval-monitor/continuous-improvement-with-production-data) for detailed steps.

### 4. Evaluate, Iterate, and Refine in Development

With production-derived data or datasets, you can now assess and improve your application in your development environment.

- **Assess Quality:** Use `mlflow.evaluate()` with your current development version of the application and the trace-based evaluation dataset. Apply relevant quality scorers and judges.
- **Analyze Results:**
  - Compare performance against the original production outputs if included in your dataset.
  - Drill down into individual examples where your development version performs poorly or differently than expected. LLM judge rationales are particularly helpful here.
  - The MLflow Evaluation UI helps visualize and compare evaluation runs.
- **Iterate on Your Application:** Use these insights to refine your prompts, models, retrieval strategies, or application logic. For example, if traces show poor relevance for queries about "Feature X," you might develop a new prompt or fine-tune a model specifically for "Feature X" queries.
- **Update Datasets:** Continuously update your development evaluation datasets with new, challenging, or representative examples from production as they emerge.

### Example: Iterating on a "Feature X" Problem

1.  **Monitor:** Production monitoring dashboards show low relevance scores and negative user feedback for queries related to "Feature X."
2.  **Select Traces:** Use `mlflow.search_traces(filter_string="inputs.query CONTAINS 'Feature X' AND (assessments.relevance_score.value < 0.5 OR assessments.user_feedback.value = 'negative')")` to find problematic traces.
3.  **Create/Augment Dataset:** Extract `inputs.query` (and `outputs.response` if available) from these traces. Create an `EvaluationDataset` named `feature_x_improvement_set`.
4.  **Develop and Evaluate:**
    - Modify your GenAI app (e.g., update a prompt, adjust RAG configuration) to improve handling of "Feature X" queries.
    - Run `mlflow.evaluate(model=your_dev_app_version, data=feature_x_improvement_set, targets="outputs.response", evaluators=["default"])`.
5.  **Analyze and Iterate:** Review the evaluation results in the MLflow UI. If scores improved and judge rationales are positive, your changes are likely beneficial. Otherwise, continue iterating.

## Preparing Data for Fine-Tuning or Retraining

Beyond evaluation, production traces, especially those with human feedback or high-quality automated assessments, can be excellent sources of data for fine-tuning your LLMs or improving your retrieval models.

- **Selection Criteria for Fine-Tuning Data:**
  - High-quality interactions where the app performed well (positive examples).
  - Interactions where the app performed poorly, along with corrected or ideal responses (negative examples with corrections). This may require a human review and labeling step. <!-- See [Human Feedback](../../human-feedback/index) for related concepts.-->
<!--
- Transform these selected traces into the format required by your fine-tuning process (e.g., prompt-completion pairs).

## Key Takeaways for Continuous Improvement

- **Iterative Cycle:** Treat GenAI app development as a continuous cycle of production monitoring, data-driven insight gathering, and development-time refinement.
- **Realism is Key:** Production traces provide the most realistic data for evaluating how your application will perform with actual users.
- **Targeted Improvement:** Focusing on problematic traces allows you to address known weaknesses systematically.
- **Combine Automated and Human Insights:** Leverage both automated assessments logged with traces and any available human feedback for a holistic view.
- **Reduces Surprises:** Testing against real-world scenarios during development helps minimize unexpected issues when you deploy new versions to production.
- **Data Quality Matters:** When curating datasets from production (especially for fine-tuning), pay attention to data quality. Human review and cleaning are often necessary.

## Prerequisites

- **Application Deployed with Tracing:** Your GenAI application deployed to a production or production-like environment with MLflow Tracing actively configured and capturing comprehensive data.
- **Access to MLflow Tracking Server:** Where trace data is being logged and is accessible.
- **Familiarity with MLflow Tools:**
  - `mlflow.search_traces()` for querying trace data.
  - `mlflow.data.EvaluationDataset` for structuring evaluation data.
  - `mlflow.evaluate()` for running evaluations.
  - [Quality Scorers and LLM Judges](/genai/eval-monitor/concepts/scorers) defined for your application's needs.
- **(For advanced analysis and alerting)** Potentially, tools or platforms for data aggregation (e.g., Spark, Pandas in a scheduled job), visualization (e.g., Databricks SQL Dashboards, Grafana), and alerting.

By thoughtfully using production data within a continuous monitoring and improvement loop, you can significantly enhance the relevance, effectiveness, and robustness of your GenAI application over time.

## Next steps

Continue your journey with these recommended actions and tutorials.

- [Build evaluation datasets](/genai/eval-monitor/build-eval-dataset) - Create test data from production traces
- [Set up quality monitoring](/genai/eval-monitor/run-scorer-in-prod) - Automatically score production traffic
- [Create custom scorers](/genai/eval-monitor/custom-scorers) - Detect specific production issues

## Reference guides

Explore detailed documentation for concepts and features mentioned in this guide.

- [Production Monitoring](/genai/eval-monitor/concepts/production-monitoring) - Understand how MLflow monitors GenAI apps in production
- [Evaluation Datasets](/genai/eval-monitor/concepts/eval-datasets) - Learn about structured test data from production traces
- [Scorers](/genai/eval-monitor/concepts/scorers) - Understand the metrics used to identify quality issues -->
