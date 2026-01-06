import json
import logging
import random
import re
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.optimizers.base import BasePromptOptimizer, _EvalFunc
from mlflow.genai.optimize.types import EvaluationResultRecord, PromptOptimizerOutput
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


# Meta-prompt template for zero-shot optimization (no evaluation data)
ZERO_SHOT_META_PROMPT_TEMPLATE = """You are an expert prompt engineer. Your task is to improve the
following prompts to achieve better performance using prompt engineering best practices.

CURRENT PROMPTS:
{current_prompts_formatted}

OBJECTIVE:
Improve these prompts to produce more accurate, helpful, and relevant outputs.

PROMPT ENGINEERING BEST PRACTICES:
1. Be specific and clear about the task and expected output format
2. Provide context and constraints when necessary
3. Use structured formatting (numbered lists, sections) for clarity
4. Include examples or demonstrations when helpful (few-shot prompting)
5. Specify the role or persona if relevant (e.g., "You are an expert...")
6. Break complex tasks into steps
7. Use explicit instructions for output format and constraints
8. Specify what to avoid or not do if important
9. Use delimiters or tags to separate different sections

TEMPLATE VARIABLES:
The following variables MUST be preserved exactly as-is in your improved prompts:
{template_variables}

{custom_guidelines}

INSTRUCTIONS:
Generate improved versions of the prompts above. Apply prompt engineering principles to improve
quality while preserving all template variables.

You must respond with a valid JSON object and nothing else. Use this exact structure:
{{
  "prompt_name_1": "improved prompt text with {{{{variables}}}} preserved",
  "prompt_name_2": "improved prompt text with {{{{variables}}}} preserved"
}}

Do not include any text before or after the JSON object. Do not include explanations or reasoning.
"""


# Meta-prompt template for few-shot optimization (with evaluation feedback)
FEW_SHOT_META_PROMPT_TEMPLATE = """You are an expert prompt engineer. Your task is to improve the
following prompts based on evaluation feedback from real examples.

CURRENT PROMPTS:
{current_prompts_formatted}

CURRENT PERFORMANCE:
Average score: {current_score:.3f}

OBJECTIVE:
Improve these prompts to produce more accurate, helpful, and relevant outputs based on the
evaluation results and feedback below.

EVALUATION EXAMPLES:
Below are examples showing how the current prompts performed. Learn from the evaluation results and
rationales.

{examples_formatted}

TEMPLATE VARIABLES:
The following variables MUST be preserved exactly as-is in your improved prompts:
{template_variables}

{custom_guidelines}

INSTRUCTIONS:
Based on the evaluation feedback above, generate improved versions of the prompts.
Analyze the examples to understand what works well and what needs improvement.

You must respond with a valid JSON object and nothing else. Use this exact structure:
{{
  "prompt_name_1": "improved prompt text with {{{{variables}}}} preserved",
  "prompt_name_2": "improved prompt text with {{{{variables}}}} preserved"
}}

Do not include any text before or after the JSON object. Do not include explanations or reasoning.
"""


@experimental(version="3.9.0")
class MetaPromptOptimizer(BasePromptOptimizer):
    """
    A prompt optimizer that uses metaprompting with LLMs to iteratively improve prompts.

    Automatically detects optimization mode based on training data:
    - Zero-shot: No evaluation data - applies general prompt engineering best practices
    - Few-shot: Has evaluation data - learns from feedback on examples

    Args:
        reflection_model: Name of the model to use for prompt optimization.
            Format: "<provider>:/<model>" (e.g., "openai:/gpt-4o",
            "anthropic:/claude-3-5-sonnet-20241022")
        num_iterations: Number of refinement iterations for few-shot mode. Zero-shot mode
            always uses a single iteration. Default: 3
        num_examples: Number of examples to randomly sample for few-shot learning. If None,
            uses all training examples. Only used when training data is provided. Default: None
        lm_kwargs: Optional dictionary of additional parameters to pass to the reflection model
            (e.g., {"temperature": 0.7, "max_tokens": 4096}). These are passed directly to
            the underlying litellm.completion() call. Default: None
        display_progress_bar: Whether to show progress bar during few-shot optimization.
            Default: False

    Example with evaluation data (few-shot mode):

        .. code-block:: python

            import mlflow
            import openai
            from mlflow.genai.optimize.optimizers import MetaPromptOptimizer
            from mlflow.genai.scorers import Correctness

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
                {"inputs": {"question": "What is 2+2?"}, "outputs": "4"},
            ]

            result = mlflow.genai.optimize_prompts(
                predict_fn=predict_fn,
                train_data=dataset,
                prompt_uris=[prompt.uri],
                optimizer=MetaPromptOptimizer(
                    reflection_model="openai:/gpt-4o",
                    num_iterations=5,
                    lm_kwargs={"temperature": 0.7, "max_tokens": 4096},
                    display_progress_bar=True,
                ),
                scorers=[Correctness(model="openai:/gpt-4o")],
                optimizer_kwargs={
                    "guidelines": "This is for a finance advisor to project tax situations. "
                    "Do not include information outside of finance."
                },
            )

            print(f"Improved prompt: {result.optimized_prompts[0].template}")

    Example without evaluation data (zero-shot mode):

        .. code-block:: python

            import mlflow
            from mlflow.genai.optimize.optimizers import MetaPromptOptimizer

            prompt = mlflow.genai.register_prompt(
                name="qa",
                template="Answer: {{question}}",
            )

            # Zero-shot mode: no evaluation data
            result = mlflow.genai.optimize_prompts(
                predict_fn=lambda question: "",  # Not used in zero-shot
                train_data=[],  # Empty dataset triggers zero-shot mode
                prompt_uris=[prompt.uri],
                optimizer=MetaPromptOptimizer(
                    reflection_model="openai:/gpt-4o",
                    num_iterations=3,
                ),
                scorers=[],  # No scorers needed for zero-shot
            )

            print(f"Improved prompt: {result.optimized_prompts[0].template}")
    """

    def __init__(
        self,
        reflection_model: str,
        num_iterations: int = 3,
        num_examples: int | None = None,
        lm_kwargs: dict[str, Any] | None = None,
        display_progress_bar: bool = False,
    ):
        from mlflow.metrics.genai.model_utils import _parse_model_uri

        self.reflection_model = reflection_model
        self.num_iterations = num_iterations
        self.num_examples = num_examples
        self.lm_kwargs = lm_kwargs or {}
        self.display_progress_bar = display_progress_bar
        self.provider, self.model = _parse_model_uri(self.reflection_model)
        self._validate_parameters()

    def _validate_parameters(self):
        if self.num_iterations < 1:
            raise MlflowException("`num_iterations` must be at least 1")

        if self.num_examples is not None and self.num_examples < 1:
            raise MlflowException("`num_examples` must be at least 1 or None")

        if not isinstance(self.lm_kwargs, dict):
            raise MlflowException("`lm_kwargs` must be a dictionary")

    def optimize(
        self,
        eval_fn: _EvalFunc,
        train_data: list[dict[str, Any]],
        target_prompts: dict[str, str],
        enable_tracking: bool = True,
        guidelines: str | None = None,
        val_data: list[dict[str, Any]] | None = None,
    ) -> PromptOptimizerOutput:
        """
        Optimize the target prompts using metaprompting.

        Automatically detects mode:
        - If train_data is empty: zero-shot mode (no evaluation)
        - If train_data has examples: few-shot mode (with evaluation)

        Args:
            eval_fn: The evaluation function that takes candidate prompts and dataset,
                returns evaluation results. Not used in zero-shot mode.
            train_data: The dataset to use for optimization. Empty list triggers zero-shot mode.
            target_prompts: The target prompt templates as dict (name -> template).
            enable_tracking: If True (default), automatically log optimization progress.
            guidelines: Optional custom instructions to guide the prompt optimization process.
                For example: "This is for a finance advisor to project tax situations.
                Do not include information outside of finance." These guidelines will be
                included in the meta-prompt sent to the reflection model.
            val_data: Optional validation dataset for evaluation. If provided, train_data is used
                for meta-prompting (showing examples to the LLM) and val_data is used for
                evaluation (measuring improvement). This prevents overfitting. If None (default),
                train_data is used for both purposes.

        Returns:
            The optimized prompts with initial and final scores (scores are None for zero-shot).
        """
        # Extract template variables
        template_variables = self._extract_template_variables(target_prompts)

        # Auto-detect mode based on training data
        if not train_data or len(train_data) == 0:
            _logger.info("No training data provided, using zero-shot metaprompting")
            return self._optimize_zero_shot(target_prompts, template_variables, guidelines)
        else:
            _logger.info(
                f"{len(train_data)} training examples provided, using few-shot metaprompting"
            )
            if val_data:
                _logger.info(
                    f"{len(val_data)} validation examples provided for evaluation"
                )
            return self._optimize_few_shot(
                eval_fn,
                train_data,
                target_prompts,
                template_variables,
                enable_tracking,
                guidelines,
                val_data,
            )

    def _optimize_zero_shot(
        self,
        target_prompts: dict[str, str],
        template_variables: dict[str, set[str]],
        guidelines: str | None = None,
    ) -> PromptOptimizerOutput:
        """
        Optimize prompts using zero-shot metaprompting (no evaluation data).

        Applies general prompt engineering best practices in a single pass.
        """
        _logger.info("Applying zero-shot prompt optimization with best practices")

        # Build meta-prompt
        meta_prompt = self._build_zero_shot_meta_prompt(
            target_prompts, template_variables, guidelines
        )

        # Call LLM to generate improved prompts
        try:
            improved_prompts = self._call_reflection_model(meta_prompt)

            # Validate template variables are preserved in improved prompts
            self._validate_template_variables(target_prompts, improved_prompts)

            _logger.info("Successfully generated improved prompts")

            return PromptOptimizerOutput(
                optimized_prompts=improved_prompts,
                initial_eval_score=None,  # No evaluation in zero-shot mode
                final_eval_score=None,
            )

        except Exception as e:
            _logger.warning(f"Zero-shot optimization failed: {e}. Returning original prompts.")
            return PromptOptimizerOutput(
                optimized_prompts=target_prompts,
                initial_eval_score=None,
                final_eval_score=None,
            )

    def _optimize_few_shot(
        self,
        eval_fn: _EvalFunc,
        train_data: list[dict[str, Any]],
        target_prompts: dict[str, str],
        template_variables: dict[str, set[str]],
        enable_tracking: bool,
        guidelines: str | None = None,
        val_data: list[dict[str, Any]] | None = None,
    ) -> PromptOptimizerOutput:
        """
        Optimize prompts using few-shot metaprompting (with evaluation feedback).

        Iteratively improves prompts based on evaluation results from training examples.

        Args:
            eval_fn: Evaluation function to score prompts
            train_data: Training data used for showing examples in meta-prompts
            target_prompts: Initial prompts to optimize
            template_variables: Template variables extracted from prompts
            enable_tracking: Whether to log metrics to MLflow
            guidelines: Optional custom guidelines for optimization
            val_data: Optional validation dataset for evaluation. If provided, train_data
                is used for meta-prompting (showing examples to the LLM) and val_data is
                used for evaluation (measuring improvement). This prevents overfitting.
                If None, train_data is used for both purposes.
        """
        # Use val_data for evaluation if provided, otherwise use train_data
        eval_data = val_data if val_data is not None else train_data

        # Baseline evaluation
        _logger.info("Evaluating baseline prompts...")
        baseline_results = eval_fn(target_prompts, eval_data)
        initial_score = self._compute_aggregate_score(baseline_results)
        _logger.info(f"Baseline score: {initial_score:.4f}")

        # Log baseline score at step 0 if tracking is enabled
        if enable_tracking:
            mlflow.log_metric("score", initial_score, step=0)

        best_prompts = target_prompts.copy()
        best_score = initial_score
        best_results = baseline_results

        # Progress bar setup
        if self.display_progress_bar:
            try:
                from tqdm import tqdm

                iterations = tqdm(
                    range(self.num_iterations),
                    desc=f"Optimizing (best: {best_score:.3f})",
                )
            except ImportError:
                iterations = range(self.num_iterations)
        else:
            iterations = range(self.num_iterations)

        for i in iterations:
            _logger.info(
                f"Few-shot iteration {i + 1}/{self.num_iterations} (current best: {best_score:.4f})"
            )

            # Sample examples for few-shot learning
            sampled_examples = self._sample_examples(train_data, best_results)

            # Build meta-prompt with evaluation feedback
            meta_prompt = self._build_few_shot_meta_prompt(
                best_prompts,
                template_variables,
                sampled_examples,
                guidelines,
            )

            # Call LLM to generate improved prompts
            try:
                improved_prompts = self._call_reflection_model(meta_prompt)

                # Validate template variables are preserved
                self._validate_template_variables(target_prompts, improved_prompts)

                # Evaluate improved prompts (using val_data if provided, otherwise train_data)
                _logger.info("Evaluating improved prompts...")
                new_results = eval_fn(improved_prompts, eval_data)
                new_score = self._compute_aggregate_score(new_results)
                _logger.info(f"New score: {new_score:.4f}")

                # Log iteration metrics if tracking is enabled
                if enable_tracking:
                    mlflow.log_metric("score", new_score, step=i + 1)
                    # Log intermediate prompts as artifacts
                    self._log_prompts_as_artifact(improved_prompts, iteration=i + 1)

                # Check if improved
                if new_score > best_score:
                    improvement = new_score - best_score
                    _logger.info(f"Improvement: +{improvement:.4f}")
                    best_prompts = improved_prompts
                    best_score = new_score
                    best_results = new_results

                    # Log improvement metric if tracking is enabled
                    if enable_tracking:
                        mlflow.log_metric("improvement", improvement, step=i + 1)
                else:
                    _logger.info("No improvement, keeping previous best")

                # Update progress bar
                if self.display_progress_bar and hasattr(iterations, "set_description"):
                    iterations.set_description(f"Optimizing (best: {best_score:.3f})")

            except Exception as e:
                _logger.warning(f"Iteration {i + 1} failed: {e}. Keeping previous best.")

        return PromptOptimizerOutput(
            optimized_prompts=best_prompts,
            initial_eval_score=initial_score,
            final_eval_score=best_score,
        )

    def _extract_template_variables(self, prompts: dict[str, str]) -> dict[str, set[str]]:
        """
        Extract template variables ({{var}}) from each prompt.

        Args:
            prompts: Dict mapping prompt_name -> template

        Returns:
            Dict mapping prompt_name -> set of variable names
        """
        variables = {}
        for name, template in prompts.items():
            # Match {{variable}} pattern (MLflow uses double braces)
            matches = re.findall(r"\{\{(\w+)\}\}", template)
            variables[name] = set(matches)
        return variables

    def _validate_template_variables(
        self, original_prompts: dict[str, str], new_prompts: dict[str, str]
    ) -> bool:
        """
        Validate that all template variables are preserved in new prompts.

        Args:
            original_prompts: Original prompt templates
            new_prompts: New prompt templates to validate

        Returns:
            True if valid

        Raises:
            MlflowException: If validation fails
        """
        original_vars = self._extract_template_variables(original_prompts)
        new_vars = self._extract_template_variables(new_prompts)

        for name in original_prompts:
            if name not in new_prompts:
                raise MlflowException(f"Prompt '{name}' missing from improved prompts")

            if original_vars[name] != new_vars[name]:
                missing = original_vars[name] - new_vars[name]
                extra = new_vars[name] - original_vars[name]
                msg = f"Template variables mismatch in prompt '{name}'."
                if missing:
                    msg += f" Missing: {missing}."
                if extra:
                    msg += f" Extra: {extra}."
                raise MlflowException(msg)

        return True

    def _log_prompts_as_artifact(self, prompts: dict[str, str], iteration: int):
        """
        Log intermediate prompts as MLflow artifacts.

        Args:
            prompts: Dict mapping prompt_name -> template
            iteration: Current iteration number
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            for prompt_name, template in prompts.items():
                # Create a filename that includes iteration and prompt name
                filename = f"iteration_{iteration:02d}_{prompt_name}.txt"
                filepath = f"{tmpdir}/{filename}"
                with open(filepath, "w") as f:
                    f.write(template)
                mlflow.log_artifact(filepath, artifact_path="intermediate_prompts")

    def _sample_examples(
        self,
        train_data: list[dict[str, Any]],
        eval_results: list[EvaluationResultRecord],
    ) -> list[tuple[dict, EvaluationResultRecord]]:
        """
        Sample examples randomly from the training data.

        If num_examples is None, returns all examples. Otherwise, randomly samples
        num_examples from the dataset.

        Args:
            train_data: Full training dataset
            eval_results: Evaluation results for current prompts

        Returns:
            List of (data_record, eval_result) tuples
        """

        data_and_eval_results = list(zip(train_data, eval_results))

        # If num_examples is None, or the number of examples is less than or equal to num_examples,
        # return all data.
        if self.num_examples is None or len(data_and_eval_results) <= self.num_examples:
            return data_and_eval_results
        # Random sample num_examples from the dataset.
        return random.sample(data_and_eval_results, self.num_examples)

    def _build_zero_shot_meta_prompt(
        self,
        current_prompts: dict[str, str],
        template_variables: dict[str, set[str]],
        guidelines: str | None = None,
    ) -> str:
        # Format the current prompts for each module
        prompts_formatted = "\n\n".join(
            [
                f"Prompt name: {name}\nTemplate: {template}"
                for name, template in current_prompts.items()
            ]
        )

        # Format template variables
        vars_formatted = "\n".join(
            [
                f"- Prompt '{name}': {', '.join(sorted(vars)) if vars else 'none'}"
                for name, vars in template_variables.items()
            ]
        )

        # Add custom guidelines to the meta-prompt if provided
        custom_guidelines = f"CUSTOM GUIDELINES:\n{guidelines}" if guidelines else ""

        return ZERO_SHOT_META_PROMPT_TEMPLATE.format(
            current_prompts_formatted=prompts_formatted,
            template_variables=vars_formatted,
            custom_guidelines=custom_guidelines,
        )

    def _build_few_shot_meta_prompt(
        self,
        current_prompts: dict[str, str],
        template_variables: dict[str, set[str]],
        sampled_examples: list[tuple[dict, EvaluationResultRecord]],
        guidelines: str | None = None,
    ) -> str:
        """Build few-shot meta-prompt with evaluation feedback."""
        # Format current prompts
        prompts_formatted = "\n\n".join(
            [
                f"Prompt name: {name}\nTemplate: {template}"
                for name, template in current_prompts.items()
            ]
        )

        # Calculate current score from sampled examples
        current_score = sum(result.score for _, result in sampled_examples) / len(sampled_examples)

        # Format examples
        examples_formatted = self._format_examples(sampled_examples)

        # Format template variables
        vars_formatted = "\n".join(
            [
                f"- Prompt '{name}': {', '.join(sorted(vars)) if vars else 'none'}"
                for name, vars in template_variables.items()
            ]
        )

        # Add custom guidelines to the meta-prompt if provided
        custom_guidelines = f"CUSTOM GUIDELINES:\n{guidelines}" if guidelines else ""

        return FEW_SHOT_META_PROMPT_TEMPLATE.format(
            current_prompts_formatted=prompts_formatted,
            current_score=current_score,
            examples_formatted=examples_formatted,
            template_variables=vars_formatted,
            custom_guidelines=custom_guidelines,
        )

    def _format_examples(self, examples: list[tuple[dict, EvaluationResultRecord]]) -> str:
        """Format examples and the evaluation results for meta-prompting."""
        formatted = []
        for i, (data, result) in enumerate(examples, 1):
            rationale_str = (
                "\n".join([f"  - {k}: {v}" for k, v in result.rationales.items()])
                if result.rationales
                else "  None"
            )

            formatted.append(
                f"""Example {i}:
  Input: {json.dumps(result.inputs)}
  Output: {result.outputs}
  Expected: {result.expectations}
  Score: {result.score:.3f}
  Rationales:
{rationale_str}
"""
            )
        return "\n".join(formatted)

    def _call_reflection_model(self, meta_prompt: str) -> dict[str, str]:
        """
        Call the reflection model to generate improved prompts.

        Args:
            meta_prompt: The meta-prompt to send

        Returns:
            Dict of improved prompts (name -> template)

        Raises:
            ImportError: If litellm is not installed
            MlflowException: If LLM call fails or response is invalid
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm is required for metaprompt optimization. "
                "Please install it with: `pip install litellm`"
            ) from e

        litellm_model = f"{self.provider}/{self.model}"

        try:
            litellm_params = {
                "model": litellm_model,
                "messages": [{"role": "user", "content": meta_prompt}],
                "response_format": {"type": "json_object"},  # Request JSON output
                "max_retries": 3,
                **self.lm_kwargs,  # Merge user-provided parameters
            }
            response = litellm.completion(**litellm_params)

            # Extract and parse response
            content = response.choices[0].message.content.strip()

            # Strip markdown code blocks if present as some models have the tendency to add them
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            content = content.removesuffix("```").strip()

            # The content should be a valid JSON object with keys being the prompt names and values
            # being the improved prompts.
            improved_prompts = json.loads(content)

            if not isinstance(improved_prompts, dict):
                raise MlflowException(
                    f"Reflection model returned invalid format. Expected JSON object, "
                    f"got {type(improved_prompts).__name__}"
                )

            for key, value in improved_prompts.items():
                if not isinstance(value, str):
                    raise MlflowException(
                        f"Prompt '{key}' must be a string, got {type(value).__name__}"
                    )

            return improved_prompts

        except json.JSONDecodeError as e:
            raise MlflowException(
                f"Failed to parse reflection model response as JSON: {e}\nResponse: {content[:500]}"
            ) from e
        except Exception as e:
            raise MlflowException(f"Failed to call reflection model {litellm_model}: {e}") from e

    def _compute_aggregate_score(self, results: list[EvaluationResultRecord]) -> float:
        """
        Compute aggregate score from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Average score across all examples
        """
        if not results:
            return 0.0

        return sum(r.score for r in results) / len(results)
