from mlflow.genai.prompts.utils import format_prompt

HALLUCINATION_FEEDBACK_NAME = "hallucination_detection"

HALLUCINATION_PROMPT_INSTRUCTIONS = """\
You are an AI assistant tasked with detecting hallucinations in AI-generated responses. \
Your job is to determine if a given response is faithful to the provided context documents, \
or if it contains hallucinated information.

A response is considered HALLUCINATED ("no") if it:
- Contains specific information not present in the provided documents
- Contradicts information in the provided documents
- Introduces new entities, facts, or claims not supported by the documents
- Makes up statistics, dates, or specific details not found in the documents

A response is considered FAITHFUL ("yes") if it:
- Only contains information supported by the provided documents
- Is a greeting or common social exchange
- Contains common knowledge or widely accepted facts
- Is a reasonable rephrasing or inference from the document content
- Expresses uncertainty or redirects the user
- Declines to answer due to insufficient information\
"""

HALLUCINATION_PROMPT = (
    HALLUCINATION_PROMPT_INSTRUCTIONS
    + """

<context>
{{context}}
</context>

<response>
{{response}}
</response>

Evaluate whether the response is faithful to the context. \
Return your assessment as JSON with the following format. \
Do not use any markdown formatting or output additional lines.
{
  "rationale": "Reason for the assessment. Start with 'Let's think step by step'",
  "result": "yes|no"
}\
"""
)


def get_prompt(*, response: str, context: str) -> str:
    return format_prompt(HALLUCINATION_PROMPT, response=response, context=context)
