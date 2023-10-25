# Location: mlflow/mlflow/metrics/genai/genai_metric.py:132
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/metrics/genai/genai_metric.py:132 '])
def test(_):
    from mlflow.metrics import EvaluationExample, make_genai_metric

    example = EvaluationExample(
        input="What is MLflow?",
        output=(
            "MLflow is an open-source platform for managing machine "
            "learning workflows, including experiment tracking, model packaging, "
            "versioning, and deployment, simplifying the ML lifecycle."
        ),
        score=4,
        justification=(
            "The definition effectively explains what MLflow is "
            "its purpose, and its developer. It could be more concise for a 5-score.",
        ),
        grading_context={
            "ground_truth": (
                "MLflow is an open-source platform for managing "
                "the end-to-end machine learning (ML) lifecycle. It was developed by "
                "Databricks, a company that specializes in big data and machine learning "
                "solutions. MLflow is designed to address the challenges that data "
                "scientists and machine learning engineers face when developing, training, "
                "and deploying machine learning models."
            )
        },
    )

    metric = make_genai_metric(
        name="correctness",
        definition=(
            "Correctness refers to how well the generated output matches "
            "or aligns with the reference or ground truth text that is considered "
            "accurate and appropriate for the given input. The ground truth serves as "
            "a benchmark against which the provided output is compared to determine the "
            "level of accuracy and fidelity."
        ),
        grading_prompt=(
            "Correctness: If the answer correctly answer the question, below "
            "are the details for different scores: "
            "- Score 0: the answer is completely incorrect, doesn’t mention anything about "
            "the question or is completely contrary to the correct answer. "
            "- Score 1: the answer provides some relevance to the question and answer "
            "one aspect of the question correctly. "
            "- Score 2: the answer mostly answer the question but is missing or hallucinating "
            "on one critical aspect. "
            "- Score 4: the answer correctly answer the question and not missing any "
            "major aspect"
        ),
        examples=[example],
        version="v1",
        model="openai:/gpt-4",
        grading_context_columns=["ground_truth"],
        parameters={"temperature": 0.0},
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
    )


if __name__ == "__main__":
    test()
