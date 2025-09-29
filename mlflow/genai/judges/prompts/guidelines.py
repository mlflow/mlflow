from mlflow.genai.prompts.utils import format_prompt

GUIDELINES_FEEDBACK_NAME = "guidelines"


GUIDELINES_PROMPT_INSTRUCTIONS = """\
Given the following set of guidelines and some inputs, please assess whether the inputs fully \
comply with all the provided guidelines. Only focus on the provided guidelines and not the \
correctness, relevance, or effectiveness of the inputs.

<guidelines>
{{guidelines}}
</guidelines>
{{guidelines_context}}\
"""

GUIDELINES_PROMPT_OUTPUT = """

Please provide your assessment using only the following json format. Do not use any markdown formatting or output additional lines. If any of the guidelines are not satisfied, the result must be "no". If none of the guidelines apply to the given inputs, the result must be "yes".
{
  "rationale": "Detailed reasoning for your assessment. If the assessment does not satisfy the guideline, state which parts of the guideline are not satisfied. Start each rationale with `Let's think step by step. `",
  "result": "yes|no"
}\
"""  # noqa: E501

GUIDELINES_PROMPT = GUIDELINES_PROMPT_INSTRUCTIONS + GUIDELINES_PROMPT_OUTPUT


def get_prompt(
    guidelines: str | list[str],
    guidelines_context: dict[str, str],
) -> str:
    if isinstance(guidelines, str):
        guidelines = [guidelines]

    return format_prompt(
        GUIDELINES_PROMPT,
        guidelines=_render_guidelines(guidelines),
        guidelines_context=_render_guidelines_context(guidelines_context),
    )


def _render_guidelines(guidelines: list[str]) -> str:
    lines = [f"<guideline>{guideline}</guideline>" for guideline in guidelines]
    return "\n".join(lines)


def _render_guidelines_context(guidelines_context: dict[str, str]) -> str:
    lines = [f"<{key}>{value}</{key}>" for key, value in guidelines_context.items()]
    return "\n".join(lines)
