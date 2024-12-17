# ruff: noqa: F821
dbutils.library.restartPython()
## Run the above in a separate cell ##

import pandas as pd

import mlflow

evals = [
    {
        "request": {
            "messages": [
                {"role": "user", "content": "How do I convert a Spark DataFrame to Pandas?"}
            ],
        },
        # Optional, needed for judging correctness.
        "expected_facts": [
            "To convert a Spark DataFrame to Pandas, you can use the toPandas() method."
        ],
    }
]

eval_result = mlflow.evaluate(
    data=pd.DataFrame.from_records(evals), model="{{ modelUri }}", model_type="databricks-agent"
)
