import time
import uuid

import pytest
from unittest.mock import patch

from mlflow.server._job_manager import PromptOptimizationJobManager
from mlflow.genai.datasets import create_dataset
from mlflow.genai.optimize.types import OptimizerOutput
from mlflow.protos.service_pb2 import GetOptimizePromptJob
from mlflow.genai.scorers import scorer
from mlflow.tracking.fluent import _get_experiment_id
from tests.genai.optimize.test_base import sample_prompt, sample_data


@pytest.fixture
def job_manager():
    """Create a fresh job manager instance for each test."""
    return PromptOptimizationJobManager()


class TestPromptOptimizationJobManager:
    """Test cases for PromptOptimizationJobManager."""

    @patch("mlflow.genai.optimize.base._DSPyMIPROv2Optimizer.optimize")
    def test_create_and_get_optimize_prompt_job(self, mock_optimizer, job_manager, sample_data, sample_prompt):
        """Test creating a prompt optimization job and then getting its info."""
        # Create and register a custom scorer
        @scorer
        def custom_accuracy(inputs, outputs, expectations):
            return 1.0
        
        # Register the custom scorer in an experiment
        custom_accuracy.register(name="custom_accuracy", experiment_id=_get_experiment_id())
        
        # Mock the optimizer to return a successful result
        def mock_fn(*args, **kwargs):
            time.sleep(1)
            return OptimizerOutput(
                final_eval_score=1.0,
                initial_eval_score=0.5,
                optimizer_name="DSPy/MIPROv2",
                optimized_prompt="optimized",
            )

        mock_optimizer.side_effect = mock_fn
        train_dataset = create_dataset(uuid.uuid4().hex)
        train_dataset.merge_records(sample_data)
        train_dataset_id = train_dataset.dataset_id
        input_prompt = f"prompts:/{sample_prompt.name}/{sample_prompt.version}"
        job_id = job_manager.create_job(
            train_dataset_id=train_dataset_id,
            eval_dataset_id=None,
            prompt_url=input_prompt,
            scorers=[
                {"name": "correctness"},
                {
                    "custom_scorer": {
                        "name": "custom_accuracy",
                        "experiment_id": _get_experiment_id()
                    }
                }
            ],
            target_llm="gpt-4",
            algorithm="DSPy/MIPROv2",
        )

        # Verify job was created successfully
        assert job_id == "0"
        assert job_id in job_manager._jobs
        
        # Wait for the job to start (it runs in a background thread)
        time.sleep(0.5)
        
        # Get the job info and verify all details after completion
        job = job_manager.get_job(job_id)
        assert job["train_dataset_id"] == train_dataset_id
        assert job["eval_dataset_id"] is None
        assert job["prompt_url"] == input_prompt
        assert job["scorers"] == [
            {"name": "correctness"},
            {
                "custom_scorer": {
                    "name": "custom_accuracy",
                    "experiment_id": _get_experiment_id(),
                }
            }
        ]
        assert job["target_llm"] == "gpt-4"
        assert job["algorithm"] == "DSPy/MIPROv2"
        assert "created_time" in job
        assert job["status"] == GetOptimizePromptJob.PromptOptimizationJobStatus.RUNNING
        assert job["result"] is None

        # Wait for the job to complete
        time.sleep(1)
        job = job_manager.get_job(job_id)
        assert job["status"] == GetOptimizePromptJob.PromptOptimizationJobStatus.COMPLETED
        assert job["result"] is not None
        assert job["result"]["prompt_url"] == f"prompts:/{sample_prompt.name}/{sample_prompt.version + 1}"
        assert job["result"]["evaluation_score"] == 1.0
        assert job["error"] is None
