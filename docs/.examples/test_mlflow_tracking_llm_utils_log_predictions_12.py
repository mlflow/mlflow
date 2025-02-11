# Location: mlflow/tracking/llm_utils.py:32
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/llm_utils.py:32 '])
def test(_):
    import mlflow

    inputs = [
        {
            "question": "How do I create a Databricks cluster with UC access?",
            "context": "Databricks clusters are ...",
        },
    ]

    outputs = [
        "<Instructions for cluster creation with UC enabled>",
    ]

    prompts = [
        "Get Databricks documentation to answer all the questions: {input}",
    ]


    with mlflow.start_run():
        # Log llm predictions
        mlflow.llm.log_predictions(inputs, outputs, prompts)


if __name__ == "__main__":
    test()
