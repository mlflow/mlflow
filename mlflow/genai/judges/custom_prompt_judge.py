import re
from difflib import unified_diff
from typing import Callable

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.judges.utils import (
    format_prompt,
    get_default_model,
    invoke_judge_model,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

_CHOICE_PATTERN = re.compile(r"\[\[([\w ]+)\]\]")


@format_docstring(_MODEL_API_DOC)
@experimental(version="3.0.0")
def custom_prompt_judge(
    *,
    name: str,
    prompt_template: str,
    numeric_values: dict[str, float] | None = None,
    model: str | None = None,
) -> Callable[..., Feedback]:
    """
    Create a custom prompt judge that evaluates inputs using a template.

    Args:
        name: Name of the judge, used as the name of returned
            :py:class:`mlflow.entities.Feedback` object.
        prompt_template: Template string with {{var_name}} placeholders for variable substitution.
            Should be prompted with choices as outputs.
        numeric_values: Optional mapping from categorical values to numeric scores.
            Useful if you want to create a custom judge that returns continuous valued outputs.
            Defaults to None.
        model: {{ model }}

    Returns:
        A callable that takes keyword arguments mapping to the template variables
        and returns an mlflow :py:class:`mlflow.entities.Feedback`.

    Example prompt template:

    .. code-block::

        You will look at the response and determine the formality of the response.

        <request>{{request}}</request>
        <response>{{response}}</response>

        You must choose one of the following categories.

        [[formal]]: The response is very formal.
        [[semi_formal]]: The response is somewhat formal. The response is somewhat formal if the
        response mentions friendship, etc.
        [[not_formal]]: The response is not formal.

    Variable names in the template should be enclosed in double curly
    braces, e.g., `{{request}}`, `{{response}}`. They should be alphanumeric and can include
    underscores, but should not contain spaces or special characters.

    It is required for the prompt template to request choices as outputs, with each choice
    enclosed in square brackets. Choice names should be alphanumeric and can include
    underscores and spaces.
    """
    model = model or get_default_model()

    if model == "databricks":
        try:
            from databricks.agents.evals.judges import custom_prompt_judge as db_custom_prompt_judge

            return db_custom_prompt_judge(
                name=name,
                prompt_template=prompt_template,
                numeric_values=numeric_values,
            )
        except ImportError:
            raise ImportError(
                "The `databricks-agents` package is required to use "
                "`mlflow.genai.judges.custom_prompt_judge` with model='databricks'. "
                "Please install it with `pip install databricks-agents`."
            )

    # Extract choices from the prompt template
    choices = _CHOICE_PATTERN.findall(prompt_template)

    if not choices:
        raise ValueError(
            "Prompt template must include choices denoted with [[CHOICE_NAME]]. "
            "No choices found in the provided prompt template."
        )

    # Validate that choices match numeric_values keys if provided
    if numeric_values is not None:
        sorted_numeric_values = sorted(numeric_values.keys())
        sorted_choices = sorted(choices)
        if sorted_numeric_values != sorted_choices:
            diff = "\n".join(
                unified_diff(
                    sorted_numeric_values,
                    sorted_choices,
                    fromfile="numeric_values_keys",
                    tofile="choices",
                )
            )
            raise ValueError(
                f"numeric_values keys must match the choices included in the prompt template.\n"
                f"numeric_values keys: {sorted_numeric_values}\n"
                f"choices in prompt: {sorted_choices}\n"
                f"Diff:\n{diff}"
            )

        # Validate that numeric_values values are numeric if provided
        if not all(isinstance(value, (int, float)) for value in numeric_values.values()):
            raise ValueError("All values in numeric_values must be numeric (int or float).")

    source = AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE,
        source_id=f"custom_prompt_judge_{name}",
    )

    def judge(**kwargs) -> Feedback:
        try:
            # Render prompt template with the given kwargs
            prompt = format_prompt(prompt_template, **kwargs)
            prompt = _remove_choice_brackets(prompt)
            prompt = _add_structured_output_instructions(prompt)

            # Call the judge
            feedback = invoke_judge_model(model, prompt, name)
            feedback.source = source

            # Feedback value must be one of the choices
            if feedback.value not in choices:
                raise ValueError(f"'{feedback.value}' is not one of the choices: {choices}")

            # Map to numeric value if mapping is provided
            if numeric_values:
                feedback.metadata = {"string_value": feedback.value}
                feedback.value = numeric_values[feedback.value]
            return feedback

        except Exception as e:
            return Feedback(name=name, source=source, error=e)

    return judge


def _add_structured_output_instructions(prompt: str) -> str:
    """Add JSON format instructions to the user prompt."""
    suffix = """
Answer ONLY in JSON and NOT in markdown, following the format:

{
    "rationale": "Reason for the decision. Start each rationale with `Let's think step by step`.",
    "result": "The category chosen."
}
"""
    return f"{prompt.strip()}\n\n{suffix}"


def _remove_choice_brackets(text: str) -> str:
    """Remove double square brackets around choices."""
    return _CHOICE_PATTERN.sub(r"\1", text)
