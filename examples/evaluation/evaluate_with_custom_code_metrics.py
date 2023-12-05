import os

import openai
import pandas as pd

import mlflow
from mlflow.metrics import make_metric
from mlflow.metrics.base import MetricValue, standard_aggregations

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."


# Helper function to check if a string is valid python code
def is_valid_python_code(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


# Create an evaluation function that iterates through the predictions
def eval_fn(predictions):
    scores = [int(is_valid_python_code(prediction)) for prediction in predictions]
    return MetricValue(
        scores=scores,
        aggregate_results=standard_aggregations(scores),
    )


# Create an EvaluationMetric object for the python code metric
valid_code_metric = make_metric(
    eval_fn=eval_fn, greater_is_better=False, name="valid_python_code", version="v1"
)

eval_df = pd.DataFrame(
    {
        "input": [
            "SELECT * FROM ",
            "import pandas",
            "def hello_world",
        ],
    }
)

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
        extra_metrics=[valid_code_metric],
    )
    print(results)

    eval_table = results.tables["eval_results_table"]
    print(eval_table)
