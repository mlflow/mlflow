import pydantic

from mlflow.genai.prompts.utils import format_prompt

SEMANTIC_MATCH_FEEDBACK_NAME = "semantic_match"


class SemanticMatchResponse(pydantic.BaseModel):
    """Structured response format for SemanticMatch scorer."""

    rationale: str = pydantic.Field(
        description="Brief explanation of the judgment comparing the two responses"
    )
    result: int = pydantic.Field(
        ge=0,
        le=100,
        description="Semantic equivalence score from 0 (completely different) to 100 (identical)",
    )


SEMANTIC_MATCH_PROMPT_INSTRUCTIONS = """\
You are a strict judge evaluating how well a student model's response \
matches a teacher model's response.

Your task is to score the student's response on multiple dimensions and compute a weighted score.

<input_context>{{input_context}}</input_context>
<expected_response>{{expected_response}}</expected_response>
<actual_response>{{actual_response}}</actual_response>

Evaluate on these criteria (each 0-100):

1. **Factual Accuracy (30%)**: Are ALL facts, numbers, names, and details exactly correct?
   - 100: Every fact matches exactly
   - 70-99: Minor factual differences that don't change meaning
   - 40-69: Some facts wrong or missing
   - 0-39: Major factual errors or contradictions

2. **Reasoning Alignment (25%)**: Does the student follow the same logical steps?
   - 100: Identical reasoning structure and intermediate steps
   - 70-99: Same approach with minor step differences
   - 40-69: Different approach but valid reasoning
   - 0-39: Flawed or missing reasoning

3. **Completeness (20%)**: Are all key points from the expected response covered?
   - 100: All points addressed with same depth
   - 70-99: All main points covered, minor omissions
   - 40-69: Some important points missing
   - 0-39: Most content missing

4. **Answer Match (15%)**: Does the final answer/conclusion match exactly?
   - 100: Exact match (ignoring trivial formatting)
   - 50: Partially correct or approximately correct
   - 0: Wrong answer or no clear answer

5. **Format Compliance (10%)**: Does the output format match the expected format?
   - 100: Same structure, formatting, notation
   - 50: Different format but readable
   - 0: Incompatible format\
"""

SEMANTIC_MATCH_PROMPT_OUTPUT = """

Compute: 0.30*factual + 0.25*reasoning + 0.20*completeness + 0.15*answer + 0.10*format

Be strict: Only give 90+ if the responses are nearly identical in meaning AND structure.
A score of 70-89 means good but noticeable differences. Below 70 means significant gaps.

Output format (JSON only, no markdown):
{"rationale": "<feedback>", "result": 65}

For the rationale, provide GENERAL feedback (not problem-specific) that could help improve \
future responses. Focus on:
- What general reasoning patterns or approaches were missing
- What type of information should be included that wasn't
- What format or structure improvements would help
Do NOT mention specific numbers, variables, or problem details - keep feedback abstract \
and applicable to similar problems.

Replace 65 with your computed weighted score (0-100).\
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
