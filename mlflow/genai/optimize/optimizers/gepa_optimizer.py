import importlib.metadata
import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging.version import Version

from mlflow.genai.optimize.optimizers.base import BasePromptOptimizer, _EvalFunc
from mlflow.genai.optimize.types import EvaluationResultRecord, PromptOptimizerOutput
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    import gepa

_logger = logging.getLogger(__name__)


@experimental(version="3.5.0")
class GepaPromptOptimizer(BasePromptOptimizer):
    """
    A prompt adapter that uses GEPA (Genetic-Pareto) optimization algorithm
    to optimize prompts.

    GEPA uses iterative mutation, reflection, and Pareto-aware candidate selection
    to improve text components like prompts. It leverages large language models to
    reflect on system behavior and propose improvements.

    Args:
        reflection_model: Name of the model to use for reflection and optimization.
            Format: "<provider>:/<model>"
            (e.g., "openai:/gpt-4o", "anthropic:/claude-3-5-sonnet-20241022").
        max_metric_calls: Maximum number of evaluation calls during optimization.
            Higher values may lead to better results but increase optimization time.
            Default: 100
        display_progress_bar: Whether to show a progress bar during optimization.
            Default: False
        gepa_kwargs: Additional keyword arguments to pass directly to
            gepa.optimize <https://github.com/gepa-ai/gepa/blob/main/src/gepa/api.py>.
            Useful for accessing advanced GEPA features not directly exposed
            through MLflow's GEPA interface.

            Note: Parameters already handled by MLflow's GEPA class will be overridden by the direct
            parameters and should not be passed through gepa_kwargs. List of predefined params:

            - max_metric_calls
            - display_progress_bar
            - seed_candidate
            - trainset
            - adapter
            - reflection_lm
            - use_mlflow

    Example:

        .. code-block:: python

            import mlflow
            import openai
            from mlflow.genai.optimize.optimizers import GepaPromptOptimizer

            prompt = mlflow.genai.register_prompt(
                name="qa",
                template="Answer the following question: {{question}}",
            )


            def predict_fn(question: str) -> str:
                completion = openai.OpenAI().chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt.format(question=question)}],
                )
                return completion.choices[0].message.content


            dataset = [
                {"inputs": {"question": "What is the capital of France?"}, "outputs": "Paris"},
                {"inputs": {"question": "What is the capital of Germany?"}, "outputs": "Berlin"},
            ]

            result = mlflow.genai.optimize_prompts(
                predict_fn=predict_fn,
                train_data=dataset,
                prompt_uris=[prompt.uri],
                optimizer=GepaPromptOptimizer(
                    reflection_model="openai:/gpt-4o",
                    display_progress_bar=True,
                ),
            )

            print(result.optimized_prompts[0].template)
    """

    def __init__(
        self,
        reflection_model: str,
        max_metric_calls: int = 100,
        display_progress_bar: bool = False,
        gepa_kwargs: dict[str, Any] | None = None,
    ):
        self.reflection_model = reflection_model
        self.max_metric_calls = max_metric_calls
        self.display_progress_bar = display_progress_bar
        self.gepa_kwargs = gepa_kwargs or {}

    def optimize(
        self,
        eval_fn: _EvalFunc,
        train_data: list[dict[str, Any]],
        target_prompts: dict[str, str],
        enable_tracking: bool = True,
    ) -> PromptOptimizerOutput:
        """
        Optimize the target prompts using GEPA algorithm.

        Args:
            eval_fn: The evaluation function that takes candidate prompts as a dict
                (prompt template name -> prompt template) and a dataset as a list of dicts,
                and returns a list of EvaluationResultRecord.
            train_data: The dataset to use for optimization. Each record should
                include the inputs and outputs fields with dict values.
            target_prompts: The target prompt templates to use. The key is the prompt template
                name and the value is the prompt template.
            enable_tracking: If True (default), automatically log optimization progress.

        Returns:
            The outputs of the prompt optimizer that includes the optimized prompts
            as a dict (prompt template name -> prompt template).
        """
        from mlflow.metrics.genai.model_utils import _parse_model_uri

        try:
            import gepa
        except ImportError as e:
            raise ImportError(
                "GEPA is not installed. Please install it with: `pip install gepa`"
            ) from e

        provider, model = _parse_model_uri(self.reflection_model)

        class MlflowGEPAAdapter(gepa.GEPAAdapter):
            """
            MLflow optimization adapter for GEPA optimization

            Args:
                eval_function: Function that evaluates candidate prompts on a dataset.
                prompts_dict: Dictionary mapping prompt names to their templates.
                tracking_enabled: Whether to log traces/metrics/params/artifacts during
                    optimization.
                full_dataset_size: Size of the full training dataset, used to distinguish
                    full validation passes from minibatch evaluations.
            """

            def __init__(self, eval_function, prompts_dict, tracking_enabled, full_dataset_size):
                self.eval_function = eval_function
                self.prompts_dict = prompts_dict
                self.prompt_names = list(prompts_dict.keys())
                self.tracking_enabled = tracking_enabled
                self.full_dataset_size = full_dataset_size
                self.validation_iteration = 0

            def evaluate(
                self,
                batch: list[dict[str, Any]],
                candidate: dict[str, str],
                capture_traces: bool = False,
            ) -> "gepa.EvaluationBatch":
                """
                Evaluate a candidate prompt using the MLflow eval function.

                Args:
                    batch: List of data instances to evaluate
                    candidate: Proposed text components (prompts)
                    capture_traces: Whether to capture execution traces.

                Returns:
                    EvaluationBatch with outputs, scores, and optional trajectories
                """
                eval_results = self.eval_function(candidate, batch)

                outputs = [result.outputs for result in eval_results]
                scores = [result.score for result in eval_results]
                trajectories = eval_results if capture_traces else None

                # Track validation candidates only during full dataset validation
                # (not during minibatch evaluation in reflective mutation)
                is_full_validation = not capture_traces and len(batch) == self.full_dataset_size
                if is_full_validation and self.tracking_enabled:
                    self._log_validation_candidate(candidate, eval_results)

                return gepa.EvaluationBatch(
                    outputs=outputs, scores=scores, trajectories=trajectories
                )

            def _log_validation_candidate(
                self,
                candidate: dict[str, str],
                eval_results: list[EvaluationResultRecord],
            ) -> None:
                """
                Log validation candidate prompts and scores as MLflow artifacts.

                Args:
                    candidate: The candidate prompts being validated
                    eval_results: Evaluation results containing scores
                """
                import mlflow

                active_run = mlflow.active_run()
                if active_run is None:
                    return

                iteration = self.validation_iteration
                self.validation_iteration += 1

                # Compute aggregate score across all records
                aggregate_score = (
                    sum(r.score for r in eval_results) / len(eval_results) if eval_results else 0.0
                )

                # Compute per-scorer average scores
                scorer_names = set()
                for result in eval_results:
                    scorer_names.update(result.individual_scores.keys())

                per_scorer_scores = {}
                for scorer_name in scorer_names:
                    scores = [
                        r.individual_scores.get(scorer_name)
                        for r in eval_results
                        if scorer_name in r.individual_scores
                    ]
                    if scores:
                        per_scorer_scores[scorer_name] = sum(scores) / len(scores)

                # Build eval results table as dict of columns
                eval_results_table = {
                    "inputs": [json.dumps(r.inputs) for r in eval_results],
                    "output": [json.dumps(r.outputs) for r in eval_results],
                    "expectation": [json.dumps(r.expectations) for r in eval_results],
                    "aggregate_score": [r.score for r in eval_results],
                }
                for scorer_name in scorer_names:
                    eval_results_table[scorer_name] = [
                        r.individual_scores.get(scorer_name) for r in eval_results
                    ]

                iteration_dir = f"prompt_candidates/iteration_{iteration}"

                # Log eval results as MLflow table
                mlflow.log_table(
                    data=eval_results_table,
                    artifact_file=f"{iteration_dir}/eval_results.json",
                )

                # Log scores summary as JSON artifact
                scores_data = {
                    "aggregate": aggregate_score,
                    "per_scorer": per_scorer_scores,
                }

                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)

                    # Write scores.json
                    scores_path = tmp_path / "scores.json"
                    with open(scores_path, "w") as f:
                        json.dump(scores_data, f, indent=2)
                    mlflow.log_artifact(str(scores_path), artifact_path=iteration_dir)

                    # Write each prompt as a separate text file
                    for prompt_name, prompt_text in candidate.items():
                        prompt_path = tmp_path / f"{prompt_name}.txt"
                        with open(prompt_path, "w") as f:
                            f.write(prompt_text)
                        mlflow.log_artifact(str(prompt_path), artifact_path=iteration_dir)

            def make_reflective_dataset(
                self,
                candidate: dict[str, str],
                eval_batch: "gepa.EvaluationBatch[EvaluationResultRecord, Any]",
                components_to_update: list[str],
            ) -> dict[str, list[dict[str, Any]]]:
                """
                Build a reflective dataset for instruction refinement.

                Args:
                    candidate: The evaluated candidate
                    eval_batch: Result of evaluate with capture_traces=True
                    components_to_update: Component names to update

                Returns:
                    Dict of reflective dataset per component
                """
                reflective_datasets = {}

                for component_name in components_to_update:
                    component_data = []

                    trajectories = eval_batch.trajectories

                    for i, (trajectory, score) in enumerate(zip(trajectories, eval_batch.scores)):
                        trace = trajectory.trace
                        spans = []
                        if trace:
                            spans = [
                                {
                                    "name": span.name,
                                    "inputs": span.inputs,
                                    "outputs": span.outputs,
                                }
                                for span in trace.data.spans
                            ]

                        component_data.append(
                            {
                                "component_name": component_name,
                                "current_text": candidate.get(component_name, ""),
                                "trace": spans,
                                "score": score,
                                "inputs": trajectory.inputs,
                                "outputs": trajectory.outputs,
                                "expectations": trajectory.expectations,
                                "rationales": trajectory.rationales,
                                "index": i,
                            }
                        )

                    reflective_datasets[component_name] = component_data

                return reflective_datasets

        adapter = MlflowGEPAAdapter(
            eval_fn, target_prompts, enable_tracking, full_dataset_size=len(train_data)
        )

        kwargs = self.gepa_kwargs | {
            "seed_candidate": target_prompts,
            "trainset": train_data,
            "adapter": adapter,
            "reflection_lm": f"{provider}/{model}",
            "max_metric_calls": self.max_metric_calls,
            "display_progress_bar": self.display_progress_bar,
            "use_mlflow": enable_tracking,
        }

        if Version(importlib.metadata.version("gepa")) < Version("0.0.18"):
            kwargs.pop("use_mlflow")
        gepa_result = gepa.optimize(**kwargs)

        optimized_prompts = gepa_result.best_candidate
        initial_score, final_score = self._extract_eval_scores(gepa_result)

        return PromptOptimizerOutput(
            optimized_prompts=optimized_prompts,
            initial_eval_score=initial_score,
            final_eval_score=final_score,
        )

    def _extract_eval_scores(self, result: "gepa.GEPAResult") -> tuple[float | None, float | None]:
        """
        Extract initial and final evaluation scores from GEPA result.

        Args:
            result: GEPA optimization result

        Returns:
            Tuple of (initial_score, final_score), both can be None if unavailable
        """
        final_score = None
        initial_score = None

        scores = result.val_aggregate_scores
        if scores and len(scores) > 0:
            # The first score is the initial baseline score
            initial_score = scores[0]
            # The highest score is the final optimized score
            final_score = max(scores)

        return initial_score, final_score
