import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from mlflow.entities.model_registry import PromptVersion
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.optimizers.base_optimizer import BasePromptOptimizer
from mlflow.genai.optimize.types import LLMParams, ObjectiveFn, OptimizerOutput
from mlflow.genai.scorers import Scorer
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    import gepa
    import pandas as pd

_logger = logging.getLogger(__name__)


@experimental(version="3.5.0")
class _GEPAOptimizer(BasePromptOptimizer):
    """
    Prompt optimizer using native GEPA (Genetic-Pareto) optimization algorithm.

    This optimizer uses GEPA directly without DSPy dependency, leveraging
    iterative mutation, reflection, and Pareto-aware candidate selection.
    """

    def optimize(
        self,
        prompt: PromptVersion,
        target_llm_params: LLMParams,
        train_data: "pd.DataFrame",
        scorers: list[Scorer],
        objective: ObjectiveFn | None = None,
        eval_data: "pd.DataFrame | None" = None,
    ) -> OptimizerOutput:
        """
        Optimize a prompt using GEPA algorithm.

        Args:
            prompt: The initial prompt version to optimize
            target_llm_params: Parameters for the target LLM
            train_data: Training dataset as a pandas DataFrame
            scorers: List of scorers to evaluate prompt performance
            objective: Optional objective function to combine scorer outputs
            eval_data: Optional evaluation dataset

        Returns:
            OptimizerOutput containing the optimized prompt and scores
        """
        try:
            import gepa
        except ImportError as e:
            raise ImportError(
                "GEPA is not installed. Please install it with: pip install gepa"
            ) from e

        _logger.info(
            f"🎯 Starting GEPA prompt optimization for: {prompt.uri}\n"
            f"⏱️ This may take several minutes or longer depending on dataset size...\n"
            f"📊 Training with {len(train_data)} examples."
        )

        # Convert DataFrame to list of dictionaries for GEPA
        # Expected format: [{"inputs": {...}, "expected_outputs": ...}, ...]
        train_list = train_data.to_dict("records")
        eval_list = eval_data.to_dict("records") if eval_data is not None else None

        # Create GEPA adapter using MLflow scorers
        adapter = self._create_gepa_adapter(
            prompt=prompt,
            target_llm_params=target_llm_params,
            scorers=scorers,
            objective=objective,
        )

        # Parse model names (convert from "provider:/model" to "provider/model")
        task_lm = self._parse_model_name(target_llm_params.model_name)

        if self.optimizer_config.optimizer_llm:
            reflection_lm = self._parse_model_name(self.optimizer_config.optimizer_llm.model_name)
        else:
            reflection_lm = task_lm

        # Prepare seed candidate - use the prompt template as the initial text
        seed_candidate = {"prompt": prompt.template}

        # Run GEPA optimization
        with self._maybe_suppress_stdout_stderr():
            result = gepa.optimize(
                seed_candidate=seed_candidate,
                trainset=train_list,
                valset=eval_list,
                adapter=adapter,
                reflection_lm=reflection_lm,
                max_metric_calls=self.optimizer_config.num_instruction_candidates * 10,
                display_progress_bar=self.optimizer_config.verbose,
            )

        optimized_template = result.best_candidate["prompt"]
        initial_score, final_score = self._extract_eval_scores(result)

        self._display_optimization_result(initial_score, final_score)

        return OptimizerOutput(
            final_eval_score=final_score,
            initial_eval_score=initial_score,
            optimized_prompt=optimized_template,
            optimizer_name="GEPA",
        )

    def _create_gepa_adapter(
        self,
        prompt: PromptVersion,
        target_llm_params: LLMParams,
        scorers: list[Scorer],
        objective: ObjectiveFn | None,
    ):
        import gepa

        class MlflowGEPAAdapter(gepa.GEPAAdapter):
            def __init__(self, prompt_version, llm_params, scorers_list, objective_fn):
                self.prompt = prompt_version
                self.llm_params = llm_params
                self.scorers = scorers_list
                self.objective = objective_fn

            def evaluate(
                self,
                batch: list[dict[str, Any]],
                candidate: dict[str, str],
                capture_traces: bool = False,
            ) -> "gepa.EvaluationBatch":
                """
                Evaluate a candidate prompt on a batch of data.

                Args:
                    batch: List of data instances with 'inputs' and 'expected_outputs'
                    candidate: Candidate text components (e.g., {"prompt": "..."})
                    capture_traces: Whether to capture execution traces

                Returns:
                    EvaluationBatch with outputs, scores, and optional trajectories
                """
                prompt_template = candidate.get("prompt", self.prompt.template)

                with ThreadPoolExecutor(
                    max_workers=MLFLOW_GENAI_EVAL_MAX_WORKERS.get(),
                    thread_name_prefix="GEPAOptimizer",
                ) as executor:
                    future_to_example = [
                        executor.submit(self._evaluate_single_example, prompt_template, record)
                        for record in batch
                    ]
                    results = [future.result() for future in future_to_example]

                outputs = [result["output"] for result in results]
                scores = [result["score"] for result in results]

                # Return GEPA's EvaluationBatch
                return gepa.EvaluationBatch(
                    outputs=outputs, scores=scores, trajectories=results if capture_traces else None
                )

            def _evaluate_single_example(
                self, prompt_template: str, record: dict[str, Any]
            ) -> dict[str, Any]:
                from mlflow.genai.prompts.utils import format_prompt

                inputs = record.get("inputs", {})
                filled_prompt = format_prompt(prompt_template, **inputs)
                expectations = record.get("expectations")
                outputs = self._call_llm(filled_prompt)
                score = self._metric(inputs, outputs, expectations, self.scorers, self.objective)

                return {
                    "inputs": inputs,
                    "output": outputs,
                    "expectations": expectations,
                    "score": score,
                }

            def _metric(
                self,
                inputs: dict[str, Any],
                outputs: dict[str, Any],
                expectations: dict[str, Any],
                scorers: list[Scorer],
                objective: ObjectiveFn | None,
            ) -> float:
                scores = {}

                for scorer in scorers:
                    scores[scorer.name] = scorer.run(
                        inputs=inputs, outputs=outputs, expectations=expectations
                    )
                if objective is not None:
                    return objective(scores)
                elif all(isinstance(score, (int, float, bool)) for score in scores.values()):
                    # Use total score by default if no objective is provided
                    return sum(scores.values())
                else:
                    non_numerical_scorers = [
                        k for k, v in scores.items() if not isinstance(v, (int, float, bool))
                    ]
                    raise MlflowException(
                        f"Scorer [{','.join(non_numerical_scorers)}] return a string, Assessment or"
                        " a list of Assessment. Please provide `objective` function to aggregate "
                        "non-numerical values into a single value for optimization."
                    )

            def _call_llm(self, prompt: str) -> str:
                try:
                    import litellm

                    response = litellm.completion(
                        model=self.llm_params.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.llm_params.temperature,
                        api_base=self.llm_params.base_uri,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    _logger.error(f"LLM call failed: {e}")
                    return ""

            def make_reflective_dataset(
                self,
                candidate: dict[str, str],
                eval_batch: "gepa.EvaluationBatch",
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

                    for i, trace in enumerate(eval_batch.trajectories):
                        component_data.append(
                            {
                                "component_name": component_name,
                                "current_text": candidate.get(component_name, ""),
                                "inputs": trace.get("inputs", {}),
                                "score": trace.get("score"),
                                "output": trace.get("output", ""),
                                "expectations": trace.get("expectations", {}),
                                "index": i,
                            }
                        )

                    reflective_datasets[component_name] = component_data

                return reflective_datasets

        return MlflowGEPAAdapter(prompt, target_llm_params, scorers, objective)

    def _extract_eval_scores(self, result: "gepa.GEPAResult") -> tuple[float | None, float | None]:
        final_score = None
        initial_score = None

        scores = result.val_aggregate_scores
        if scores and len(scores) > 0:
            # The first score is the initial baseline score
            initial_score = scores[0]
            # The highest score is the final optimized score
            final_score = max(scores)

        return initial_score, final_score

    def _display_optimization_result(self, initial_score: float | None, final_score: float | None):
        if final_score is None:
            return

        if initial_score is not None:
            if abs(initial_score - final_score) < 0.0001:
                _logger.info(f"Optimization complete! Score remained stable at: {final_score:.4f}.")
            else:
                improvement = final_score - initial_score
                _logger.info(
                    f"🎉 Optimization complete! "
                    f"Initial score: {initial_score:.4f}. "
                    f"Final score: {final_score:.4f} "
                    f"(+{improvement:.4f})."
                )
        else:
            _logger.info(f"Optimization complete! Final score: {final_score:.4f}.")

    def _parse_model_name(self, model_name: str) -> str:
        # Convert "provider:/model" to "provider/model"
        # TODO: Use parse_model_name util after merging the model switch branch
        if ":" in model_name:
            model_name = model_name.replace(":/", "/")
        return model_name
