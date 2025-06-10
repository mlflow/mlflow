---
description: >
  Learn how to integrate MLflow evaluations into your CI/CD pipeline to automate quality checks, prevent regressions, and ensure consistent GenAI application performance before deployment.
last_update:
  date: 2025-05-18
---

# Evaluate within a CI/CD Pipeline

<!--
:::danger UNDER CONSTRUCTION
Flesh this page out
:::

Integrating automated quality evaluation into your CI/CD pipeline ensures your GenAI applications maintain consistent quality as they evolve. This guide shows you how to set up automated evaluations using GitHub Actions or Databricks deployment jobs.

**What you'll learn:**

- Set up GitHub Actions workflow for automated evaluation
- Configure Databricks deployment jobs for model lifecycle management
- Define quality gates to prevent regressions
- Deploy apps with confidence using automated testing

## Why CI/CD evaluation matters

Automated evaluation in your CI/CD pipeline provides:

- **Regression prevention**: Catch quality degradations before production
- **Consistent standards**: Enforce quality thresholds across all deployments
- **Automated decisions**: Pass/fail builds based on objective metrics
- **Deployment confidence**: Know your app meets quality requirements

## Prerequisites

- GenAI application instrumented with [MLflow Tracing](/mlflow3/genai/tracing/index)
- [Evaluation datasets](/genai/eval-monitor/build-eval-dataset) stored in Unity Catalog
- [Scorers](/genai/eval-monitor/concepts/scorers) defined for your quality metrics
- Git repository containing your app code and scorers

## GitHub Actions workflow

### Step 1: Prepare your repository structure

Organize your repository to support CI/CD evaluation:

```
your-repo/
├── app/
│   ├── __init__.py
│   └── main.py          # Your GenAI app with MLflow tracing
├── scorers/
│   ├── __init__.py
│   └── custom_scorers.py # Your custom scorers
├── evaluation/
│   ├── __init__.py
│   └── run_evaluation.py # Evaluation script
├── requirements.txt
└── .github/
    └── workflows/
        └── evaluate.yml  # GitHub Actions workflow
```

### Step 2: Create the evaluation script

Create `evaluation/run_evaluation.py` to run evaluations and check quality gates:

```python
import os
import sys
import mlflow
import mlflow.genai.datasets
from mlflow.genai.scorers import (
    RetrievalGroundedness,
    RelevanceToQuery,
    Safety,
    Guidelines,
)
from typing import Dict, Any

# Add parent directory to path to import your app and scorers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import generate_sales_email  # Your GenAI app
from scorers.custom_scorers import BusinessToneScorer  # Your custom scorers


def load_evaluation_dataset(uc_table_name: str):
    """Load evaluation dataset from Unity Catalog"""
    return mlflow.genai.datasets.get_dataset(uc_table_name)


def run_evaluation(app_version: str, dataset_name: str) -> Dict[str, Any]:
    """Run evaluation and return results"""

    # Load evaluation dataset
    eval_dataset = load_evaluation_dataset(dataset_name)

    # Configure scorers
    scorers = [
        RetrievalGroundedness(),
        RelevanceToQuery(),
        Safety(),
        Guidelines(
            name="follows_instructions",
            guidelines="The generated response must follow the user's instructions.",
        ),
        BusinessToneScorer(),  # Custom scorer from your repo
    ]

    # Run evaluation
    eval_results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=generate_sales_email,
        scorers=scorers,
    )

    # Tag the run with version info
    mlflow.set_tags({
        "ci_run": "true",
        "app_version": app_version,
        "github_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "github_ref": os.environ.get("GITHUB_REF", "unknown"),
    })

    return eval_results


def check_quality_gates(eval_results) -> bool:
    """Check if evaluation results meet quality thresholds"""

    # Define quality gates
    quality_gates = {
        "retrieval_groundedness_value": {"threshold": 0.90, "comparison": ">="},
        "relevance_to_query_value": {"threshold": 0.85, "comparison": ">="},
        "safety_value": {"threshold": 0.95, "comparison": ">="},
        "follows_instructions_value": {"threshold": 0.80, "comparison": ">="},
        "business_tone_score_value": {"threshold": 0.75, "comparison": ">="},
    }

    all_passed = True
    failures = []

    # Check each metric against its threshold
    for metric_name, gate in quality_gates.items():
        if metric_name not in eval_results.metrics:
            print(f"⚠️  Warning: Metric '{metric_name}' not found in results")
            continue

        score = eval_results.metrics[metric_name]["value"]
        threshold = gate["threshold"]

        if gate["comparison"] == ">=":
            passed = score >= threshold
        elif gate["comparison"] == "<=":
            passed = score <= threshold
        else:
            passed = False

        if passed:
            print(f"✅ {metric_name}: {score:.3f} (threshold: {threshold})")
        else:
            all_passed = False
            failures.append(f"{metric_name}: {score:.3f} < {threshold}")
            print(f"❌ {metric_name}: {score:.3f} (threshold: {threshold})")

    if failures:
        print("\n🚫 Quality gate failures:")
        for failure in failures:
            print(f"   - {failure}")

    return all_passed


def compare_with_baseline(eval_results, baseline_run_id: str = None):
    """Compare results with baseline version (optional)"""
    if not baseline_run_id:
        return

    try:
        baseline_run = mlflow.get_run(baseline_run_id)
        print("\n📊 Comparison with baseline:")

        for metric_name in eval_results.metrics:
            current = eval_results.metrics[metric_name]["value"]
            baseline = baseline_run.data.metrics.get(metric_name)

            if baseline:
                diff = current - baseline
                pct_change = (diff / baseline) * 100
                symbol = "📈" if diff > 0 else "📉"
                print(f"   {symbol} {metric_name}: {current:.3f} ({pct_change:+.1f}%)")

    except Exception as e:
        print(f"⚠️  Could not compare with baseline: {e}")


def main():
    """Main evaluation entry point"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--app-version", required=True, help="App version being tested")
    parser.add_argument("--dataset", required=True, help="UC table name for eval dataset")
    parser.add_argument("--baseline-run-id", help="Optional baseline run ID for comparison")
    parser.add_argument("--staging-endpoint", help="Optional staging endpoint to test")
    args = parser.parse_args()

    print(f"🧪 Running evaluation for version: {args.app_version}")
    print(f"📊 Using dataset: {args.dataset}")

    # Option to test staging endpoint instead of local code
    if args.staging_endpoint:
        print(f"🔗 Testing staging endpoint: {args.staging_endpoint}")
        from mlflow.genai import to_predict_fn
        predict_fn = to_predict_fn(f"endpoints:/{args.staging_endpoint}")

        # Run evaluation with endpoint
        eval_results = mlflow.genai.evaluate(
            data=load_evaluation_dataset(args.dataset),
            predict_fn=predict_fn,
            scorers=[
                RetrievalGroundedness(),
                RelevanceToQuery(),
                Safety(),
            ],
        )
    else:
        # Run evaluation with local code
        eval_results = run_evaluation(args.app_version, args.dataset)

    # Check quality gates
    gates_passed = check_quality_gates(eval_results)

    # Optional: Compare with baseline
    if args.baseline_run_id:
        compare_with_baseline(eval_results, args.baseline_run_id)

    # Exit with appropriate code
    if gates_passed:
        print("\n✅ All quality gates passed!")
        mlflow.set_tag("ci_quality_check", "passed")
        sys.exit(0)
    else:
        print("\n❌ Quality gates failed!")
        mlflow.set_tag("ci_quality_check", "failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 3: Create GitHub Actions workflow

Create `.github/workflows/evaluate.yml`:

```yaml
name: GenAI App Evaluation

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Configure MLflow and Databricks
        env:
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
        run: |
          # Set environment variables for MLflow
          echo "MLFLOW_TRACKING_URI=databricks" >> $GITHUB_ENV
          echo "MLFLOW_EXPERIMENT_ID=${{ secrets.MLFLOW_EXPERIMENT_ID }}" >> $GITHUB_ENV
          echo "DATABRICKS_TOKEN=$DATABRICKS_TOKEN" >> $GITHUB_ENV
          echo "DATABRICKS_HOST=$DATABRICKS_HOST" >> $GITHUB_ENV

      - name: Run evaluation
        id: evaluate
        run: |
          python evaluation/run_evaluation.py \
            --app-version "${{ github.sha }}" \
            --dataset "${{ vars.EVAL_DATASET_TABLE }}" \
            --baseline-run-id "${{ vars.BASELINE_RUN_ID }}"

      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            // Read evaluation summary if your script writes one
            const summary = `
            ## 🧪 GenAI Evaluation Results

            **Status**: ${{ steps.evaluate.outcome == 'success' && '✅ Passed' || '❌ Failed' }}
            **Version**: \`${{ github.sha }}\`

            View detailed results in [MLflow](${{ secrets.DATABRICKS_HOST }}/ml/experiments/${{ secrets.MLFLOW_EXPERIMENT_ID }})
            `;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: summary
            });

      - name: Deploy to staging (if main branch)
        if: github.ref == 'refs/heads/main' && success()
        run: |
          echo "Deploying to staging endpoint..."
          # Add your deployment logic here
```

### Step 4: Configure GitHub repository

Add the following secrets to your GitHub repository:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add these repository secrets:

   - `DATABRICKS_TOKEN`: Your Databricks personal access token
   - `DATABRICKS_HOST`: Your Databricks workspace URL
   - `MLFLOW_EXPERIMENT_ID`: Your MLflow experiment ID

3. Add these repository variables:
   - `EVAL_DATASET_TABLE`: Unity Catalog table name (e.g., `catalog.schema.eval_dataset`)
   - `BASELINE_RUN_ID`: (Optional) MLflow run ID of production baseline

## Databricks deployment jobs

For teams using Databricks, deployment jobs provide an integrated solution for model lifecycle management with built-in approvals and governance.

### Step 1: Create evaluation notebook

Create an evaluation notebook that deployment jobs will run:

```python
# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation for Deployment Job

# COMMAND ----------

# Install required packages
%pip install --upgrade "mlflow[databricks]>=3.1.0"
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.genai.datasets
from mlflow.genai.scorers import (
    RetrievalGroundedness,
    RelevanceToQuery,
    Safety,
    Guidelines,
)
import json

# Get job parameters
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
eval_dataset_name = dbutils.widgets.get("eval_dataset", "catalog.schema.eval_dataset")

print(f"Evaluating model: {model_name} version {model_version}")
print(f"Using dataset: {eval_dataset_name}")

# COMMAND ----------

# Load the model version
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

# Create predict function wrapper
def predict_fn(**inputs):
    """Wrapper to ensure model returns dict format"""
    result = model.predict(inputs)
    if isinstance(result, dict):
        return result
    return {"response": result}

# COMMAND ----------

# Load evaluation dataset
eval_dataset = mlflow.genai.datasets.get_dataset(eval_dataset_name)

# Configure scorers
scorers = [
    RetrievalGroundedness(),
    RelevanceToQuery(),
    Safety(),
    Guidelines(
        name="follows_instructions",
        guidelines="The response must follow user instructions precisely.",
    ),
]

# COMMAND ----------

# Run evaluation
eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=scorers,
    model_id=model_uri,  # Link results to model version
)

print("Evaluation complete!")

# COMMAND ----------

# Check quality gates
quality_gates = {
    "retrieval_groundedness_value": 0.90,
    "relevance_to_query_value": 0.85,
    "safety_value": 0.95,
    "follows_instructions_value": 0.80,
}

gates_passed = True
failures = []

for metric_name, threshold in quality_gates.items():
    if metric_name in eval_results.metrics:
        score = eval_results.metrics[metric_name]["value"]
        if score < threshold:
            gates_passed = False
            failures.append(f"{metric_name}: {score:.3f} < {threshold}")
            print(f"❌ {metric_name}: {score:.3f} (threshold: {threshold})")
        else:
            print(f"✅ {metric_name}: {score:.3f} (threshold: {threshold})")

# COMMAND ----------

# Write evaluation summary
summary = {
    "model_name": model_name,
    "model_version": model_version,
    "gates_passed": gates_passed,
    "metrics": {k: v["value"] for k, v in eval_results.metrics.items()},
    "failures": failures,
}

# Tag the model version with results
client = mlflow.tracking.MlflowClient(registry_uri="databricks-uc")
client.set_model_version_tag(
    name=model_name,
    version=model_version,
    key="evaluation_status",
    value="passed" if gates_passed else "failed"
)

# COMMAND ----------

if not gates_passed:
    raise Exception(f"Quality gates failed: {failures}")

print("✅ All quality gates passed! Model ready for approval.")
```

### Step 2: Create deployment job programmatically

Create a notebook to set up the deployment job:

```python
# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Create Deployment Job for GenAI Model

# COMMAND ----------

%pip install databricks-sdk --upgrade
dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
import os

# Initialize client
w = WorkspaceClient()

# REQUIRED: Update these values
MODEL_NAME = "catalog.schema.your_genai_model"
EVAL_DATASET = "catalog.schema.eval_dataset"
SERVING_ENDPOINT_NAME = "your-genai-endpoint"
SERVICE_PRINCIPAL_ID = "your-service-principal-id"  # Recommended for security

# COMMAND ----------

# Define job configuration
job_config = jobs.JobSettings(
    name=f"GenAI Deployment Job - {MODEL_NAME}",
    description="Automated evaluation, approval, and deployment for GenAI model",

    # Job parameters
    parameters=[
        jobs.JobParameter(name="model_name", default=MODEL_NAME),
        jobs.JobParameter(name="model_version", default="1"),
        jobs.JobParameter(name="eval_dataset", default=EVAL_DATASET),
    ],

    # Run as service principal for security
    run_as=jobs.JobRunAs(
        service_principal_name=SERVICE_PRINCIPAL_ID
    ),

    # Prevent concurrent runs
    max_concurrent_runs=1,

    # Define tasks
    tasks=[
        # Task 1: Evaluation
        jobs.Task(
            task_key="evaluation",
            description="Evaluate model quality",
            notebook_task=jobs.NotebookTask(
                notebook_path="/Workspace/deployment_jobs/evaluation",
                base_parameters={
                    "model_name": "{{job.parameters.model_name}}",
                    "model_version": "{{job.parameters.model_version}}",
                    "eval_dataset": "{{job.parameters.eval_dataset}}",
                }
            ),
            job_cluster_key="serverless",
        ),

        # Task 2: Approval
        jobs.Task(
            task_key="approval_check",
            description="Wait for human approval",
            depends_on=[jobs.TaskDependency(task_key="evaluation")],
            notebook_task=jobs.NotebookTask(
                notebook_path="/Workspace/deployment_jobs/approval",
                base_parameters={
                    "model_name": "{{job.parameters.model_name}}",
                    "model_version": "{{job.parameters.model_version}}",
                }
            ),
            job_cluster_key="serverless",
            retry_on_timeout=False,  # Don't retry approval
            max_retries=0,
        ),

        # Task 3: Deployment
        jobs.Task(
            task_key="deployment",
            description="Deploy to serving endpoint",
            depends_on=[jobs.TaskDependency(task_key="approval_check")],
            notebook_task=jobs.NotebookTask(
                notebook_path="/Workspace/deployment_jobs/deployment",
                base_parameters={
                    "model_name": "{{job.parameters.model_name}}",
                    "model_version": "{{job.parameters.model_version}}",
                    "endpoint_name": SERVING_ENDPOINT_NAME,
                }
            ),
            job_cluster_key="serverless",
        ),
    ],

    # Use serverless compute
    job_clusters=[
        jobs.JobCluster(
            job_cluster_key="serverless",
            new_cluster=jobs.ClusterSpec(
                spark_version="",  # Empty for serverless
                node_type_id="",   # Empty for serverless
                num_workers=0,     # 0 for serverless
                data_security_mode=jobs.DataSecurityMode.SINGLE_USER,
            )
        )
    ],
)

# COMMAND ----------

# Create the job
created_job = w.jobs.create(**job_config.as_dict())
print(f"✅ Created deployment job: {created_job.job_id}")
print(f"View job at: {w.config.host}/#job/{created_job.job_id}")

# COMMAND ----------

# Connect job to model
import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")

try:
    # Update existing model
    client.update_registered_model(
        name=MODEL_NAME,
        deployment_job_id=str(created_job.job_id)
    )
    print(f"✅ Connected deployment job to model: {MODEL_NAME}")
except mlflow.exceptions.RestException:
    # Create new model with deployment job
    client.create_registered_model(
        name=MODEL_NAME,
        deployment_job_id=str(created_job.job_id)
    )
    print(f"✅ Created model with deployment job: {MODEL_NAME}")
```

### Step 3: Trigger and monitor deployment

Once connected, the deployment job automatically triggers when new model versions are created. You can also manually trigger it:

```python
# Register a new model version (triggers deployment job automatically)
import mlflow

with mlflow.start_run() as run:
    # Log your model
    mlflow.pyfunc.log_model(
        "model",
        python_model=YourGenAIModel(),
        registered_model_name=MODEL_NAME,
    )

print("✅ Model registered - deployment job will start automatically")
```

## Best practices

### Organize your scorers

Keep scorers in version control so they're available in CI/CD:

```python
# scorers/custom_scorers.py
from mlflow.genai.scorers import scorer
from typing import Dict, Any

@scorer
def business_tone_scorer(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Check if response maintains professional business tone"""
    response = outputs.get("response", "")

    # Your scoring logic
    informal_phrases = ["hey", "gonna", "wanna", "yeah"]
    score = 1.0
    for phrase in informal_phrases:
        if phrase.lower() in response.lower():
            score -= 0.25

    return {
        "score": max(0, score),
        "reasoning": f"Business tone score based on formality check"
    }
```

### Version your evaluation datasets

Store evaluation datasets in Unity Catalog for versioning and governance:

```python
# Create versioned evaluation dataset
import mlflow.genai.datasets

# Create from production traces
dataset = mlflow.genai.datasets.create_dataset(
    uc_table_name="catalog.schema.eval_dataset_v2",
)

# Add version tag
dataset.add_tag("version", "2.0")
dataset.add_tag("created_from", "production_traces_2024_01")
```

### Test locally before CI/CD

Run evaluations locally to debug before pushing:

```bash
# Set environment variables
export DATABRICKS_TOKEN="your-token"
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export MLFLOW_TRACKING_URI="databricks"
export MLFLOW_EXPERIMENT_ID="your-experiment-id"

# Run evaluation
python evaluation/run_evaluation.py \
  --app-version "local-test" \
  --dataset "catalog.schema.eval_dataset"
```

## Troubleshooting

### Common issues

**Authentication errors in GitHub Actions:**

- Verify secrets are correctly set in repository settings
- Check token permissions include MLflow API access
- Ensure workspace URL includes `https://`

**Scorers not found:**

- Ensure scorer modules are in the repository
- Check import paths in evaluation script
- Verify `sys.path` includes parent directory

**Deployment job approval stuck:**

- Approval tasks always fail first time (expected behavior)
- Click "Approve" in model version UI to continue
- Check user has APPLY TAG permission on model

## Next steps

Continue your journey with these recommended actions and tutorials.

- [Build evaluation datasets](/genai/eval-monitor/build-eval-dataset) - Create regression test sets from production
- [Set up production monitoring](/genai/eval-monitor/run-scorer-in-prod) - Use the same scorers in production
- [Track application versions](/mlflow3/genai/prompt-version-mgmt/version-tracking/track-application-versions-with-mlflow) - Version your GenAI apps

## Reference guides

Explore detailed documentation for concepts and features mentioned in this guide.

- [Evaluation Harness](/genai/eval-monitor/concepts/eval-harness) - Understand evaluation orchestration
- [Scorers](/genai/eval-monitor/concepts/scorers) - Learn about quality metrics used in CI/CD
- [App Version Tracking](/mlflow3/genai/prompt-version-mgmt/version-tracking/version-concepts) - Deployment versioning concepts -->
