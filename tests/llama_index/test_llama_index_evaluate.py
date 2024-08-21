import pandas as pd
import pytest

import mlflow
from mlflow.metrics import latency


@pytest.mark.parametrize(
    "data",
    [
        # Single row
        pd.DataFrame(
            {
                "inputs": ["What is MLflow?"],
                "ground_truth": ["What is MLflow?"],
            }
        ),
        # Multiple rows
        pd.DataFrame(
            {
                "inputs": ["What is MLflow?", "What is Spark?"],
                "ground_truth": ["What is MLflow?", "Not what is Spark?"],
            }
        ),
    ],
)
@pytest.mark.parametrize("engine_type", ["query", "chat"])
def test_llama_index_evaluate(data, engine_type, single_index):
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            llama_index_model=single_index,
            engine_type=engine_type,
            artifact_path="llama_index",
        )

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    eval_dataset = mlflow.data.from_pandas(data, targets="ground_truth")

    with mlflow.start_run():
        eval_result = mlflow.evaluate(
            pyfunc_model,
            data=eval_dataset,
            extra_metrics=[latency()],
        )
    assert eval_result.metrics["latency/mean"] > 0
