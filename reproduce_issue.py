
import pandas as pd
import mlflow
from mlflow.metrics.genai import make_genai_metric
from mlflow.metrics.genai import model_utils
from unittest import mock
import time

# Mock response
MOCK_RESPONSE = """
{
  "score": 5,
  "justification": "Excellent response."
}
"""

def mock_score_model_on_payload(*args, **kwargs):
    time.sleep(0.1) # Simulate network latency
    return MOCK_RESPONSE

def reproduce():
    # Define a custom metric
    metric = make_genai_metric(
        name="correctness",
        definition="Correctness of the answer",
        grading_prompt="Score 1-5",
        model="openai:/gpt-4",
        parameters={"temperature": 0.0},
        grading_context_columns=["targets"],
        aggregations=["mean"]
    )

    # Create dummy data
    n_samples = 10
    data = pd.DataFrame({
        "inputs": [f"Question {i}" for i in range(n_samples)],
        "predictions": [f"Answer {i}" for i in range(n_samples)],
        "targets": [f"Target {i}" for i in range(n_samples)]
    })

    # Mock the model call
    with mock.patch("mlflow.metrics.genai.model_utils.score_model_on_payload", side_effect=mock_score_model_on_payload) as mock_model:
        start_time = time.time()
        
        # Run evaluation
        results = metric.eval_fn(
            data["predictions"],
            {}, # metrics
            data["inputs"],
            data["targets"]
        )
        
        end_time = time.time()
        
        print(f"Processed {n_samples} samples in {end_time - start_time:.2f} seconds")
        print(f"Call count: {mock_model.call_count}")
        assert mock_model.call_count == n_samples
        print("Reproduction successful: Individual calls were made.")

if __name__ == "__main__":
    reproduce()
