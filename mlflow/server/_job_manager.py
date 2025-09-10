import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mlflow.environment_variables import MLFLOW_MAX_CONCURRENT_PROMPT_OPTIMIZATION_JOBS
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.types import LLMParams, OptimizerConfig
from mlflow.genai.scorers.builtin_scorers import get_builtin_scorer_by_name
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.service_pb2 import GetOptimizePromptJob

_logger = logging.getLogger(__name__)


# Prompt Optimization Job Management
class PromptOptimizationJobManager:
    """Manages prompt optimization jobs using a thread pool for controlled concurrency."""

    def __init__(self):
        self._jobs: dict[str, dict[str, Any]] = {}
        self._next_job_id = 0
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=MLFLOW_MAX_CONCURRENT_PROMPT_OPTIMIZATION_JOBS.get(),
            thread_name_prefix="prompt_opt",
        )

    def create_job(
        self,
        *,
        train_dataset_id: str,
        eval_dataset_id: str | None,
        prompt_url: str,
        scorers: list[Any],
        target_llm: str,
        algorithm: str | None,
    ) -> str:
        """Create a new prompt optimization job."""
        with self._lock:
            job_id = str(self._next_job_id)
            self._next_job_id += 1

            self._jobs[job_id] = {
                "status": GetOptimizePromptJob.PromptOptimizationJobStatus.PENDING,
                "train_dataset_id": train_dataset_id,
                "eval_dataset_id": eval_dataset_id,
                "prompt_url": prompt_url,
                "scorers": scorers,
                "target_llm": target_llm,
                "algorithm": algorithm,
                "result": None,
                "error": None,
                "created_time": int(time.time() * 1000),
            }

            # Submit the job to the thread pool
            self._executor.submit(self._run_job, job_id)

            return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get job status and result."""
        job = self._jobs.get(job_id)
        if job:
            return job.copy()
        return None

    def _run_job(self, job_id: str):
        """Run the prompt optimization job."""
        from mlflow.genai.datasets import get_dataset
        from mlflow.genai.optimize import optimize_prompt
        from mlflow.genai.scorers import get_scorer

        try:
            job = self._jobs[job_id]
            job["status"] = GetOptimizePromptJob.PromptOptimizationJobStatus.RUNNING

            # Load datasets
            train_dataset = get_dataset(dataset_id=job["train_dataset_id"])
            train_df = train_dataset.to_df()

            if eval_dataset_id := job.get("eval_dataset_id"):
                eval_dataset = get_dataset(dataset_id=eval_dataset_id)
                eval_df = eval_dataset.to_df()
            else:
                eval_df = None

            # Validate dataset structure
            required_columns = ["inputs", "expectations"]
            missing_columns = []
            for col in required_columns:
                if col not in train_df.columns:
                    missing_columns.append(col)

            if missing_columns:
                raise MlflowException(
                    f"Training dataset missing required columns: {missing_columns}. "
                    f"Available columns: {list(train_df.columns)}",
                    INVALID_PARAMETER_VALUE,
                )

            # Get scorers
            scorer_instances = []
            for scorer_param in job["scorers"]:
                try:
                    if "name" in scorer_param:
                        # Built-in scorer by name
                        scorer_name = scorer_param["name"]
                        scorer = get_builtin_scorer_by_name(scorer_name)
                        scorer_instances.append(scorer)
                    elif "custom_scorer" in scorer_param:
                        # Custom scorer - need to load from experiment
                        custom_scorer = scorer_param["custom_scorer"]
                        scorer = get_scorer(
                            name=custom_scorer["name"],
                            experiment_id=custom_scorer["experiment_id"],
                            version=custom_scorer.get("version"),
                        )
                        scorer_instances.append(scorer)
                    else:
                        raise MlflowException(
                            f"Invalid scorer parameter: {scorer_param}",
                            INVALID_PARAMETER_VALUE,
                        )
                except Exception as e:
                    raise MlflowException(
                        f"Failed to create scorer from parameter '{scorer_param}': {e}",
                        INVALID_PARAMETER_VALUE,
                    )

            if not scorer_instances:
                raise MlflowException(
                    "No valid scorers provided for optimization",
                    INVALID_PARAMETER_VALUE,
                )

            # Log optimization parameters
            _logger.info(f"Starting prompt optimization job: {job_id}")

            # Set up LLM parameters

            # Parse target_llm to create LLMParams
            target_llm = job["target_llm"]
            llm_params = LLMParams(model_name=target_llm)

            # Set up optimizer config
            algorithm = job["algorithm"]

            algorithm_kwarg = {"algorithm": algorithm} if algorithm else {}

            optimizer_config = OptimizerConfig(**algorithm_kwarg)

            prompt_input_url = job["prompt_url"]

            # Call the actual optimize_prompt function
            result = optimize_prompt(
                target_llm_params=llm_params,
                prompt=prompt_input_url,
                train_data=train_df,
                scorers=scorer_instances,
                eval_data=eval_df,
                optimizer_config=optimizer_config,
            )

            _logger.info(f"Prompt optimization job {job_id} completed.")

            # Save optimization result
            job["result"] = {
                "prompt_url": result.prompt.uri,
                "evaluation_score": result.final_eval_score,
            }
            job["status"] = GetOptimizePromptJob.PromptOptimizationJobStatus.COMPLETED

        except Exception as e:
            job["status"] = GetOptimizePromptJob.PromptOptimizationJobStatus.FAILED
            job["error"] = str(e)
            _logger.error(f"Prompt optimization job {job_id} failed: {e}")


# Global job manager instance
_prompt_optimization_job_manager = PromptOptimizationJobManager()
