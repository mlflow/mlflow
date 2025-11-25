"""Manual test script for DeepEval integration with MLflow."""

import os

from mlflow.genai.scorers.deepeval import get_judge, AnswerRelevancy

# Test basic functionality with a simple metric
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY not set")


judge = get_judge("AnswerRelevancy", threshold=0.7, model="openai/gpt-4o-mini")
feedback = judge(
    inputs="What is MLflow?",
    outputs="MLflow is an open source platform for managing the ML lifecycle.",
)
print(f"  Result: {feedback.value}, Score: {feedback.metadata.get('score')}")
