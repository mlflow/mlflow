export const getDatasetCodeSnippet = (experimentId: string, scorersDocLink?: string) => `import mlflow
import os
from mlflow.genai import evaluate
from mlflow.genai.scorers import (
    Safety,
    RelevanceToQuery,
    Guidelines,
)

os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your API key
mlflow.set_experiment(experiment_id="${experimentId}")

# Step 1: Define evaluation dataset
eval_dataset = [{
  "inputs": {
    "query": "What is MLflow?",
  }
}]

# Step 2: Define predict_fn
# predict_fn will be called for every row in your evaluation
# dataset. Replace with your app's prediction function.
# NOTE: The **kwargs to predict_fn are the same as the keys of
# the \`inputs\` in your dataset.
def predict(query):
  return query + " an answer"

# Step 3: Run evaluation
# Select scorers relevant to your use case.${scorersDocLink ? `\n# See all available scorers: ${scorersDocLink}` : ''}
evaluate(
  data=eval_dataset,
  predict_fn=predict,
  scorers=[
    Safety(),
    RelevanceToQuery(),
    Guidelines(name="conciseness", guidelines="Responses must be concise."),
  ],
)

# Results will appear back in this UI`;

export const getTraceCodeSnippet = (experimentId: string) => `import mlflow
import os
from mlflow.genai import evaluate
from mlflow.genai.scorers import (
    Safety,
    RelevanceToQuery,
    Guidelines,
)

os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your API key
mlflow.set_experiment(experiment_id="${experimentId}")

# Step 1: Pull traces to evaluate.
# Adjust max_results, or add a filter_string for time/status, etc.
# See: https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/
traces = mlflow.search_traces(max_results=20)

# Step 2: Run evaluation. No predict_fn needed — inputs/outputs
# are extracted from the trace objects automatically.
evaluate(
  data=traces,
  scorers=[
    Safety(),
    RelevanceToQuery(),
    Guidelines(name="conciseness", guidelines="Responses must be concise."),
  ],
)

# Results will appear back in this UI`;
