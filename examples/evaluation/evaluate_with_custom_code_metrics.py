import logging
import os

import openai
import pandas as pd
from datasets import load_dataset

import mlflow
from mlflow.metrics import make_metric
from mlflow.metrics.base import MetricValue
from mlflow.metrics.metric_definitions import standard_aggregations

logging.getLogger("mlflow").setLevel(logging.ERROR)

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."


# Create an evaluation function that iterates through the predictions
def eval_fn(predictions, targets, metrics, prompt, test, entry_point, task_id):
    scores = []
    for i in range(len(predictions)):
        problem_dict = {
            "prompt": prompt[i],
            "test": test[i],
            "entry_point": entry_point[i],
            "task_id": task_id[i],
        }
        result = check_correctness(problem_dict, predictions[0], 3.0)
        print(result)
        if result["passed"]:
            scores.append(1)
        else:
            scores.append(0)
    return MetricValue(
        scores=scores,
        aggregate_results=standard_aggregations(scores),
    )


# Create an EvaluationMetric object for the python code metric
passing_code_metric = make_metric(
    eval_fn=eval_fn, greater_is_better=False, name="passing_code_metric", version="v1"
)

# Load the openai_humaneval dataset
humaneval_ds = load_dataset("openai_humaneval")
test_data_df = humaneval_ds["test"].to_pandas()
eval_df = test_data_df.head()

with mlflow.start_run() as run:
    system_prompt = (
        "Generate code that is less than 50 characters. Return only python code and nothing else."
    )
    logged_model = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{question}"},
        ],
    )

    results = mlflow.evaluate(
        logged_model.model_uri,
        eval_df,
        model_type="text",
        extra_metrics=[passing_code_metric],
        evaluators="default",
        evaluator_config={
            "default": {
                "col_mapping": {"input": "prompt"},
            }
        },
    )
    print(results.metrics)

    eval_table = results.table["eval_results_table"]
    print(eval_table)
