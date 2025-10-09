from typing import TYPE_CHECKING, Any

from mlflow.genai.optimize.optimizers.base import BasePromptOptimizer, _EvalFunc
from mlflow.genai.optimize.types import EvaluationResultRecord, LLMParams, PromptAdapterOutput
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    import gepa


@experimental(version="3.5.0")
class GepaPromptOptimizer(BasePromptOptimizer):
    """
    A prompt adapter that uses GEPA (Genetic-Pareto) optimization algorithm
    to optimize prompts.

    GEPA uses iterative mutation, reflection, and Pareto-aware candidate selection
    to improve text components like prompts. It leverages large language models to
    reflect on system behavior and propose improvements.

    Args:
        max_metric_calls: Maximum number of evaluation calls during optimization.
            Higher values may lead to better results but increase optimization time.
            Default: 100
        reflection_lm: Optional LLM model name for the reflection model.
            This should be a stronger model used to reflect on and propose improvements.
            Format: "<provider>/<model>" (e.g., "openai/gpt-4",
            "anthropic/claude-3-5-sonnet-20241022").
            If not provided, the task LLM will be used for reflection.
            Default: None
        display_progress_bar: Whether to show a progress bar during optimization.
            Default: False

    Example:

        .. code-block:: python

            import mlflow
            import openai
            from mlflow.genai.optimize import LLMParams
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
                target_prompt_uris=[prompt.uri],
                optimizer_lm_params=LLMParams(model_name="openai:/gpt-4o"),
                optimizer=GepaPromptOptimizer(display_progress_bar=True),
            )

            print(result.optimized_prompts[0].template)
    """

    def __init__(
        self,
        max_metric_calls: int = 100,
        reflection_lm: str | None = None,
        display_progress_bar: bool = False,
    ):
        self.max_metric_calls = max_metric_calls
        self.reflection_lm = reflection_lm
        self.display_progress_bar = display_progress_bar

    def optimize(
        self,
        eval_fn: _EvalFunc,
        train_data: list[dict[str, Any]],
        target_prompts: dict[str, str],
        optimizer_lm_params: LLMParams,
    ) -> PromptAdapterOutput:
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
            optimizer_lm_params: The optimizer LLM parameters to use.

        Returns:
            The outputs of the prompt adapter that includes the optimized prompts
            as a dict (prompt template name -> prompt template).
        """
        from mlflow.genai.optimize.optimizers.utils import parse_model_name

        try:
            import gepa
        except ImportError as e:
            raise ImportError(
                "GEPA is not installed. Please install it with: pip install gepa"
            ) from e

        model_name = parse_model_name(optimizer_lm_params.model_name)

        class MlflowGEPAAdapter(gepa.GEPAAdapter):
            def __init__(self, eval_function, prompts_dict):
                self.eval_function = eval_function
                self.prompts_dict = prompts_dict
                self.prompt_names = list(prompts_dict.keys())

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
                    capture_traces: Whether to capture execution traces

                Returns:
                    EvaluationBatch with outputs, scores, and optional trajectories
                """
                eval_results = self.eval_function(candidate, batch)

                outputs = [result.outputs for result in eval_results]
                scores = [result.score for result in eval_results]
                trajectories = eval_results if capture_traces else None

                return gepa.EvaluationBatch(
                    outputs=outputs, scores=scores, trajectories=trajectories
                )

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
                        trace_str = trajectory.trace.to_json() if trajectory.trace else ""
                        component_data.append(
                            {
                                "component_name": component_name,
                                "current_text": candidate.get(component_name, ""),
                                "trace": trace_str,
                                "score": score,
                                "expectations": trajectory.expectations,
                                "index": i,
                            }
                        )

                    reflective_datasets[component_name] = component_data

                return reflective_datasets

        adapter = MlflowGEPAAdapter(eval_fn, target_prompts)

        reflection_lm = self.reflection_lm if self.reflection_lm else model_name

        gepa_result = gepa.optimize(
            seed_candidate=target_prompts,
            trainset=train_data,
            adapter=adapter,
            reflection_lm=reflection_lm,
            max_metric_calls=self.max_metric_calls,
            display_progress_bar=self.display_progress_bar,
        )

        optimized_prompts = gepa_result.best_candidate
        initial_score, final_score = self._extract_eval_scores(gepa_result)

        return PromptAdapterOutput(
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
