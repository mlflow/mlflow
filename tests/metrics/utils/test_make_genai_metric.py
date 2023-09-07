import re

import pandas as pd
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.metrics.base import EvaluationExample
from mlflow.metrics.utils.make_genai_metric import _format_variable_string, make_genai_metric


# Create a custom mock class for MyModel
class FakeModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return pd.DataFrame(
            {
                "Score": [3],
                "Justification": [
                    "The definition effectively explains what MLflow is "
                    "its purpose, and its developer. It could be more concise for a 5-score."
                ],
            }
        )


def test_make_genai_metric_pyfunc_success():
    with mlflow.start_run():
        llm_model_info = mlflow.pyfunc.log_model(
            "model",
            python_model=FakeModel(),
        )

    example = EvaluationExample(
        input="What is MLflow?",
        output="MLflow is an open-source platform for managing machine "
        "learning workflows, including experiment tracking, model packaging, "
        "versioning, and deployment, simplifying the ML lifecycle.",
        score=4,
        justification="The definition effectively explains what MLflow is "
        "its purpose, and its developer. It could be more concise for a 5-score.",
        variables={
            "ground_truth": "MLflow is an open-source platform for managing "
            "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
            "a company that specializes in big data and machine learning solutions. MLflow is "
            "designed to address the challenges that data scientists and machine learning "
            "engineers face when developing, training, and deploying machine learning models."
        },
    )

    custom_metric = make_genai_metric(
        name="correctness",
        version="v1",
        definition="Correctness refers to how well the generated output matches "
        "or aligns with the reference or ground truth text that is considered "
        "accurate and appropriate for the given input. The ground truth serves as "
        "a benchmark against which the provided output is compared to determine the "
        "level of accuracy and fidelity.",
        grading_prompt="Correctness: If the answer correctly answer the question, below are the "
        "details for different scores: "
        "- Score 0: the answer is completely incorrect, doesn’t mention anything about "
        "the question or is completely contrary to the correct answer. "
        "- Score 1: the answer provides some relevance to the question and answer one aspect "
        "of the question correctly. "
        "- Score 2: the answer mostly answer the question but is missing or hallucinating on one "
        "critical aspect. "
        "- Score 4: the answer correctly answer the question and not missing any major aspect",
        examples=[example],
        model=llm_model_info.model_uri,
        variables=["ground_truth"],
        parameters={"temperature": 1.0},
        greater_is_better=True,
        aggregations=["mean", "variance", "p90"],
    )

    eval_df = pd.DataFrame(
        {
            "input": ["What is MLflow?"],
            "prediction": [
                "MLflow is an open-source platform for managing machine "
                "learning workflows, including experiment tracking, model packaging, "
                "versioning, and deployment, simplifying the ML lifecycle."
            ],
            "ground_truth": [
                "MLflow is an open-source platform for managing "
                "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
                "a company that specializes in big data and machine learning solutions. MLflow is "
                "designed to address the challenges that data scientists and machine learning "
                "engineers face when developing, training, and deploying machine learning models."
            ],
        }
    )

    metric_value = custom_metric.eval_fn(eval_df)

    assert metric_value.scores == [3]
    assert metric_value.justifications == [
        "The definition effectively "
        "explains what MLflow is its purpose, and its developer. It could be "
        "more concise for a 5-score."
    ]

    assert metric_value.aggregate_results == {
        "mean": 3,
        "variance": 0,
        "p90": 3,
    }


def test_make_genai_metric_failure():
    example = EvaluationExample(
        input="input",
        output="output",
        score=4,
        justification="justification",
        variables={"ground_truth": "ground_truth"},
    )
    import pandas as pd

    eval_df = pd.DataFrame(
        {
            "input": ["What is MLflow?"],
            "prediction": ["predictions"],
            "ground_truth": ["truth"],
        }
    )

    custom_metric1 = make_genai_metric(
        name="correctness",
        version="v-latest",
        definition="definition",
        grading_prompt="grading_prompt",
        examples=[example],
        model="model",
        variables=["ground_truth"],
        parameters={"temperature": 1.0},
        greater_is_better=True,
        aggregations=["mean"],
    )
    with pytest.raises(
        MlflowException,
        match=re.escape(
            "Failed to find evaluation model for version v-latest."
            "Please check the correctness of the version"
        ),
    ):
        custom_metric1.eval_fn(eval_df)

    custom_metric2 = make_genai_metric(
        name="correctness",
        version="v1",
        definition="definition",
        grading_prompt="grading_prompt",
        examples=[example],
        model="model",
        variables=["ground_truth-error"],
        parameters={"temperature": 1.0},
        greater_is_better=True,
        aggregations=["mean"],
    )
    with pytest.raises(
        MlflowException,
        match=re.escape(
            "ground_truth-error does not exist in the Eval DataFrame "
            "Index(['input', 'prediction', 'ground_truth'], dtype='object')."
        ),
    ):
        custom_metric2.eval_fn(eval_df)

    with mlflow.start_run():
        llm_model_info = mlflow.pyfunc.log_model(
            "model",
            python_model=FakeModel(),
        )
    custom_metric3 = make_genai_metric(
        name="correctness",
        version="v1",
        definition="definition",
        grading_prompt="grading_prompt",
        examples=[example],
        model=llm_model_info.model_uri,
        variables=["ground_truth"],
        parameters={"temperature": 1.0},
        greater_is_better=True,
        aggregations=["random-fake"],
    )
    with pytest.raises(
        MlflowException,
        match=re.escape("Invalid aggregate option random-fake"),
    ):
        custom_metric3.eval_fn(eval_df)


def test_format_variable_string():
    variable_string = _format_variable_string(
        ["foo", "bar"], pd.DataFrame({"foo": ["foo"], "bar": ["bar"]}), 0
    )

    assert variable_string == "Provided foo: foo\nProvided bar: bar"

    with pytest.raises(
        MlflowException,
        match=re.escape(
            "bar does not exist in the Eval DataFrame " "Index(['foo'], dtype='object')."
        ),
    ):
        variable_string = _format_variable_string(["foo", "bar"], pd.DataFrame({"foo": ["foo"]}), 0)
