import json
import logging
import re
from contextlib import nullcontext
from typing import Any

import mlflow
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.optimizers.base import BasePromptOptimizer, _EvalFunc
from mlflow.genai.optimize.types import EvaluationResultRecord, PromptOptimizerOutput
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)

# Compiled regex pattern for extracting template variables
_TEMPLATE_VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")


# Unified meta-prompt template that supports both zero-shot and few-shot modes
META_PROMPT_TEMPLATE = """\
You are an expert prompt engineer. Your task is to improve
the following prompts to achieve better performance.

CURRENT PROMPTS:
{current_prompts_formatted}

{evaluation_examples}

PROMPT ENGINEERING BEST PRACTICES:
Apply these proven techniques to create effective prompts:

1. **Clarity & Specificity**: Be explicit about the task, expected output format,
and any constraints
2. **Structured Formatting**: Use numbered lists, sections, or delimiters to
organize complex instructions clearly
3. **Few-Shot Examples**: Include concrete examples showing desired input/output
pairs when appropriate
4. **Role/Persona**: Specify expertise level if relevant (e.g., "You are an expert
mathematician...")
5. **Step-by-Step Decomposition**: Break complex reasoning tasks into explicit
steps or phases
6. **Output Format Specification**: Explicitly define the format, structure, and
constraints for outputs
7. **Constraint Specification**: Clearly state what to avoid, exclude, or not do
8. **Verification Instructions**: Add self-checking steps for calculation-heavy
or error-prone tasks
9. **Chain-of-Thought Prompting**: For reasoning tasks, explicitly instruct to
show intermediate steps

CRITICAL REQUIREMENT - TEMPLATE VARIABLES:
The following variables MUST be preserved EXACTLY as shown in the original prompts.
DO NOT modify, remove, or change the formatting of these variables in any way:
{template_variables}

IMPORTANT: Template variables use double curly braces like {{{{variable_name}}}}.
You MUST copy them exactly as they appear in the original prompt into your improved
prompt. If a variable appears as {{{{question}}}} in the original, it must appear as
{{{{question}}}} in your improvement.

{custom_guidelines}

INSTRUCTIONS:
Generate improved versions of the prompts by applying relevant prompt engineering
best practices. Make your prompts specific and actionable.

{extra_instructions}

CRITICAL: Preserve all template variables in their exact original format with
double curly braces.

CRITICAL: You must respond with a valid JSON object using the EXACT prompt names
shown above. The JSON keys must match the "Prompt name" fields exactly. Use this
structure:
{{
{response_format_example}
}}

REMINDER:
1. Use the exact prompt names as JSON keys (e.g., if the prompt is named
"aime_solver", use "aime_solver" as the key)
2. Every template variable from the original prompt must appear unchanged in your
improved version
3. Apply best practices that are most relevant to the task at hand

Do not include any text before or after the JSON object. Do not include
explanations or reasoning.
"""


@experimental(version="3.9.0")
class MetaPromptOptimizer(BasePromptOptimizer):
    """
    A prompt optimizer that uses metaprompting with LLMs to improve prompts in a single pass.

    Automatically detects optimization mode based on training data:
    - Zero-shot: No evaluation data - applies general prompt engineering best practices
    - Few-shot: Has evaluation data - learns from evaluation results

    This optimizer performs a single optimization pass, making it faster than iterative
    approaches like GEPA while requiring less data. The optimized prompt is always
    registered regardless of performance improvement.

    Args:
        reflection_model: Name of the model to use for prompt optimization.
            Format: "<provider>:/<model>" (e.g., "openai:/gpt-5.2",
            "anthropic:/claude-sonnet-4-5-20250929")
        lm_kwargs: Optional dictionary of additional parameters to pass to the reflection
            model (e.g., {"temperature": 1.0, "max_tokens": 4096}). These are passed
            directly to the underlying litellm.completion() call. Default: None
        guidelines: Optional custom guidelines to provide domain-specific or task-specific
            context for prompt optimization (e.g., "This is for a finance advisor to
            project tax situations."). Default: None

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
                prompt = mlflow.genai.load_prompt("prompts:/qa@latest")
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
                    lm_kwargs={"temperature": 1.0, "max_tokens": 4096},
                ),
                scorers=[Correctness(model="openai:/gpt-4o")],
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
                    guidelines="This is for a finance advisor to project tax situations.",
                ),
                scorers=[],  # No scorers needed for zero-shot
            )

            print(f"Improved prompt: {result.optimized_prompts[0].template}")
    """

    def __init__(
        self,
        reflection_model: str,
        lm_kwargs: dict[str, Any] | None = None,
        guidelines: str | None = None,
    ):
        from mlflow.metrics.genai.model_utils import _parse_model_uri

        self.reflection_model = reflection_model
        self.lm_kwargs = lm_kwargs or {}
        self.guidelines = guidelines
        self.provider, self.model = _parse_model_uri(self.reflection_model)
        self._validate_parameters()

    def _validate_parameters(self):
        if not isinstance(self.lm_kwargs, dict):
            raise MlflowException("`lm_kwargs` must be a dictionary")

    def optimize(
        self,
        eval_fn: _EvalFunc,
        train_data: list[dict[str, Any]],
        target_prompts: dict[str, str],
        enable_tracking: bool = True,
    ) -> PromptOptimizerOutput:
        """
        Optimize the target prompts using metaprompting in a single pass.

        Automatically detects mode:
        - If train_data is empty: zero-shot mode (no evaluation)
        - If train_data has examples: few-shot mode (with baseline evaluation for feedback)

        The optimized prompt is always returned regardless of performance improvement.

        Args:
            eval_fn: The evaluation function that takes candidate prompts and dataset,
                returns evaluation results. Not used in zero-shot mode.
            train_data: The dataset to use for optimization. Empty list triggers zero-shot
                mode. In few-shot mode, train_data is always used for baseline evaluation
                (to capture feedback) and for showing examples in the meta-prompt.
            target_prompts: The target prompt templates as dict (name -> template).
            enable_tracking: If True (default), automatically log optimization progress.

        Returns:
            The optimized prompts with initial score (final_eval_score is None for
            single-pass).
        """
        # Extract template variables
        template_variables = self._extract_template_variables(target_prompts)

        # Auto-detect mode based on training data
        if not train_data:
            _logger.info("No training data provided, using zero-shot metaprompting")
            return self._optimize_zero_shot(target_prompts, template_variables, enable_tracking)
        else:
            _logger.info(
                f"{len(train_data)} training examples provided, using few-shot metaprompting"
            )

            return self._optimize_few_shot(
                eval_fn,
                train_data,
                target_prompts,
                template_variables,
                enable_tracking,
            )

    def _optimize_zero_shot(
        self,
        target_prompts: dict[str, str],
        template_variables: dict[str, set[str]],
        enable_tracking: bool,
    ) -> PromptOptimizerOutput:
        """
        Optimize prompts using zero-shot metaprompting (no evaluation data).

        Applies general prompt engineering best practices in a single pass.
        """
        _logger.info("Applying zero-shot prompt optimization with best practices")

        meta_prompt = self._build_zero_shot_meta_prompt(target_prompts, template_variables)

        try:
            improved_prompts = self._call_reflection_model(meta_prompt, enable_tracking)

            self._validate_prompt_names(target_prompts, improved_prompts)
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
    ) -> PromptOptimizerOutput:
        """
        Optimize prompts using few-shot metaprompting (with evaluation feedback).

        Performs a single optimization pass based on evaluation results from training examples.
        The optimized prompt is always returned regardless of performance improvement.

        Args:
            eval_fn: Evaluation function to score prompts
            train_data: Training data used for baseline evaluation to capture feedback
            target_prompts: Initial prompts to optimize
            template_variables: Template variables extracted from prompts
            enable_tracking: Whether to log metrics to MLflow
        """
        # Always evaluate baseline on train_data to capture feedback for metaprompting
        _logger.info("Evaluating baseline prompts on training data...")
        baseline_results = eval_fn(target_prompts, train_data)
        initial_score = self._compute_aggregate_score(baseline_results)
        if initial_score is not None:
            _logger.info(f"Baseline score: {initial_score:.4f}")

        # Build meta-prompt with evaluation feedback
        _logger.info("Generating optimized prompts...")
        meta_prompt = self._build_few_shot_meta_prompt(
            target_prompts,
            template_variables,
            baseline_results,
        )
        # Call LLM to generate improved prompts
        try:
            improved_prompts = self._call_reflection_model(meta_prompt, enable_tracking)

            self._validate_prompt_names(target_prompts, improved_prompts)
            self._validate_template_variables(target_prompts, improved_prompts)

            _logger.info("Successfully generated optimized prompts")

        except Exception as e:
            _logger.warning(f"Few-shot optimization failed: {e}. Returning original prompts.")
            return PromptOptimizerOutput(
                optimized_prompts=target_prompts,
                initial_eval_score=initial_score,
                final_eval_score=None,
            )

        final_score = None
        if initial_score is not None:
            _logger.info(
                "Evaluating optimized prompts on training data, please note that this is more of "
                "a sanity check than a final evaluation because the data has already been used "
                "for meta-prompting. To accurately evaluate the optimized prompts, please use a "
                "separate validation dataset and run mlflow.genai.evaluate() on it."
            )
            final_results = eval_fn(improved_prompts, train_data)
            final_score = self._compute_aggregate_score(final_results)
            _logger.info(f"Final score: {final_score:.4f}")

        return PromptOptimizerOutput(
            optimized_prompts=improved_prompts,
            initial_eval_score=initial_score,
            final_eval_score=final_score,
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
            matches = _TEMPLATE_VAR_PATTERN.findall(template)
            variables[name] = set(matches)
        return variables

    def _validate_prompt_names(
        self, original_prompts: dict[str, str], new_prompts: dict[str, str]
    ) -> bool:
        """
        Validate that prompt names match between original and new prompts.

        Args:
            original_prompts: Original prompt templates
            new_prompts: New prompt templates to validate

        Returns:
            True if valid

        Raises:
            MlflowException: If prompt names don't match
        """
        # Check for unexpected prompts in the improved prompts
        if unexpected_prompts := set(new_prompts) - set(original_prompts):
            raise MlflowException(
                f"Unexpected prompts found in improved prompts: {sorted(unexpected_prompts)}"
            )

        # Check for missing prompts in the improved prompts
        if missing_prompts := set(original_prompts) - set(new_prompts):
            raise MlflowException(
                f"Prompts missing from improved prompts: {sorted(missing_prompts)}"
            )

        return True

    def _validate_template_variables(
        self, original_prompts: dict[str, str], new_prompts: dict[str, str]
    ) -> bool:
        """Validate that all template variables are preserved in new prompts."""
        original_vars = self._extract_template_variables(original_prompts)
        new_vars = self._extract_template_variables(new_prompts)

        for name in original_prompts:
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

    def _build_zero_shot_meta_prompt(
        self,
        current_prompts: dict[str, str],
        template_variables: dict[str, set[str]],
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
        custom_guidelines = f"CUSTOM GUIDELINES:\n{self.guidelines}" if self.guidelines else ""

        # Format example JSON response with actual prompt names
        response_format_example = "\n".join(
            [
                f'  "{name}": "improved prompt text with variables preserved exactly"'
                for name in current_prompts.keys()
            ]
        )

        return META_PROMPT_TEMPLATE.format(
            current_prompts_formatted=prompts_formatted,
            evaluation_examples="",
            extra_instructions="",
            template_variables=vars_formatted,
            custom_guidelines=custom_guidelines,
            response_format_example=response_format_example,
        )

    def _build_few_shot_meta_prompt(
        self,
        current_prompts: dict[str, str],
        template_variables: dict[str, set[str]],
        eval_results: list[EvaluationResultRecord],
    ) -> str:
        """Build few-shot meta-prompt with evaluation feedback."""
        # Format current prompts
        prompts_formatted = "\n\n".join(
            [
                f"Prompt name: {name}\nTemplate: {template}"
                for name, template in current_prompts.items()
            ]
        )

        if not eval_results:
            raise MlflowException(
                "Few-shot metaprompting requires evaluation results. "
                "No evaluation results were provided to _build_few_shot_meta_prompt."
            )

        # Calculate current score from evaluation results (if scores are available)
        current_score = self._compute_aggregate_score(eval_results)

        # Format examples and their evaluation results in the meta-prompt
        examples_formatted = self._format_examples(eval_results)

        # Format template variables
        vars_formatted = "\n".join(
            [
                f"- Prompt '{name}': {', '.join(sorted(vars)) if vars else 'none'}"
                for name, vars in template_variables.items()
            ]
        )

        # Add custom guidelines to the meta-prompt if provided
        custom_guidelines = f"CUSTOM GUIDELINES:\n{self.guidelines}" if self.guidelines else ""

        # Format example JSON response with actual prompt names
        response_format_example = "\n".join(
            [
                f'  "{name}": "improved prompt text with variables preserved exactly"'
                for name in current_prompts.keys()
            ]
        )

        # Build evaluation examples section (with or without score)
        if current_score is not None:
            score_info = f" (Current Score: {current_score:.3f})"
            analysis_instructions = """
Before applying best practices, analyze the examples to identify:
1. **Common Failure Patterns**: What mistakes appear repeatedly? (wrong format,
   missing steps, calculation errors, etc.)
2. **Success Patterns**: What made successful examples work? (format, detail level,
   reasoning approach)
3. **Key Insights**: What do the rationales tell you about quality criteria and
   needed improvements?
4. **Task Requirements**: What output format, explanation level, and edge cases
   are expected?"""
        else:
            score_info = ""
            analysis_instructions = """
Before applying best practices, analyze the examples to identify:
1. **Output Patterns**: What are the expected outputs for different inputs?
2. **Task Requirements**: What output format, explanation level, and edge cases
   are expected?
3. **Common Themes**: What patterns do you see in the input-output relationships?"""

        evaluation_examples = f"""EVALUATION EXAMPLES{score_info}:
Below are examples showing how the current prompts performed. Study these to identify
patterns in what worked and what failed.

{examples_formatted}
{analysis_instructions}"""

        extra_instructions = """
Focus on applying best practices that directly address the observed patterns.
Add specific instructions, format specifications, or verification steps that would
improve the prompt's effectiveness."""

        return META_PROMPT_TEMPLATE.format(
            current_prompts_formatted=prompts_formatted,
            evaluation_examples=evaluation_examples,
            extra_instructions=extra_instructions,
            template_variables=vars_formatted,
            custom_guidelines=custom_guidelines,
            response_format_example=response_format_example,
        )

    def _format_examples(self, eval_results: list[EvaluationResultRecord]) -> str:
        """Format evaluation results for meta-prompting."""
        formatted = []
        for i, result in enumerate(eval_results, 1):
            rationale_str = (
                "\n".join([f"  - {k}: {v}" for k, v in result.rationales.items()])
                if result.rationales
                else "  None"
            )

            # Build example with optional score
            example_lines = [
                f"Example {i}:",
                f"  Input: {json.dumps(result.inputs)}",
                f"  Output: {result.outputs}",
                f"  Expected: {result.expectations}",
            ]
            if result.score is not None:
                example_lines.append(f"  Score: {result.score:.3f}")
                example_lines.append(f"  Rationales:\n{rationale_str}")

            formatted.append("\n".join(example_lines) + "\n")
        return "\n".join(formatted)

    def _call_reflection_model(
        self, meta_prompt: str, enable_tracking: bool = True
    ) -> dict[str, str]:
        """Call the reflection model to generate improved prompts."""
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm is required for metaprompt optimization. "
                "Please install it with: `pip install litellm`"
            ) from e

        litellm_model = f"{self.provider}/{self.model}"

        litellm_params = {
            "model": litellm_model,
            "messages": [{"role": "user", "content": meta_prompt}],
            "response_format": {"type": "json_object"},  # Request JSON output
            "max_retries": 3,
            **self.lm_kwargs,  # Merge user-provided parameters
        }

        content = None  # Initialize to avoid NameError in exception handler

        span_context = (
            mlflow.start_span(name="metaprompt_reflection", span_type=SpanType.LLM)
            if enable_tracking
            else nullcontext()
        )

        with span_context as span:
            if enable_tracking:
                span.set_inputs({"meta_prompt": meta_prompt, "model": litellm_model})

            try:
                response = litellm.completion(**litellm_params)

                # Extract and parse response
                content = response.choices[0].message.content.strip()

                # Strip markdown code blocks if present as some models have the tendency to add them
                if content.startswith("```json"):
                    content = content[7:]
                elif content.startswith("```"):
                    content = content[3:]
                content = content.removesuffix("```").strip()

                # The content should be a valid JSON object with keys being the prompt
                # names and values being the improved prompts.
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

                if enable_tracking:
                    span.set_outputs(improved_prompts)

                return improved_prompts

            except json.JSONDecodeError as e:
                response_preview = content[:2000] if content else "No content received"
                raise MlflowException(
                    f"Failed to parse reflection model response as JSON: {e}\n"
                    f"Response: {response_preview}"
                ) from e
            except Exception as e:
                raise MlflowException(
                    f"Failed to call reflection model {litellm_model}: {e}"
                ) from e

    def _compute_aggregate_score(self, results: list[EvaluationResultRecord]) -> float | None:
        """
        Compute aggregate score from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Average score across all examples, or None if no results or scores are None
        """
        if not results:
            return None

        # If any score is None, return None (no scorers were provided)
        scores = [r.score for r in results]
        if any(s is None for s in scores):
            return None

        return sum(scores) / len(scores)
