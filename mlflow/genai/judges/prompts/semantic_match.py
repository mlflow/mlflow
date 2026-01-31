from mlflow.genai.prompts.utils import format_prompt

SEMANTIC_MATCH_FEEDBACK_NAME = "semantic_match"

SEMANTIC_MATCH_PROMPT_INSTRUCTIONS = """\
You are an expert judge evaluating whether two responses are semantically equivalent.

Given a question/prompt and two responses (the expected response from a teacher model \
and the actual response from a student model), determine if they convey the same meaning \
and information.

Consider:
1. Core meaning and intent - Do both responses answer the question the same way?
2. Key information - Are all important facts/details present in both?
3. Reasoning quality - If reasoning is shown, is the logic equivalent?
4. Minor differences in wording, style, or formatting should NOT affect the score \
if the meaning is preserved.

<input_context>{{input_context}}</input_context>
<expected_response>{{expected_response}}</expected_response>
<actual_response>{{actual_response}}</actual_response>\
"""

SEMANTIC_MATCH_PROMPT_OUTPUT = """

Please evaluate semantic equivalence using only the following json format. \
Do not use any markdown formatting or output additional lines.
{
  "rationale": "Brief explanation of your judgment comparing the two responses. \
Start with `Let's think step by step`",
  "result": "yes|no"
}

Use "yes" if the responses are semantically equivalent (same meaning, key facts present).
Use "no" if there are meaningful differences in content, facts, or reasoning.\
"""

SEMANTIC_MATCH_PROMPT = SEMANTIC_MATCH_PROMPT_INSTRUCTIONS + SEMANTIC_MATCH_PROMPT_OUTPUT


def get_prompt(
    input_context: str,
    expected_response: str,
    actual_response: str,
) -> str:
    """Generate semantic match evaluation prompt.

    Args:
        input_context: The input/question context
        expected_response: The expected response (from teacher model)
        actual_response: The actual response to evaluate (from student model)

    Returns:
        Formatted prompt string
    """
    return format_prompt(
        SEMANTIC_MATCH_PROMPT,
        input_context=input_context,
        expected_response=expected_response,
        actual_response=actual_response,
    )
