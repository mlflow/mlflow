from mlflow.genai.prompts.utils import format_prompt

RETRIEVAL_RELEVANCE_PROMPT = """\
Consider the following question and document. You must determine whether the document provides information that is (fully or partially) relevant to the question. Do not focus on the correctness or completeness of the document. Do not make assumptions, approximations, or bring in external knowledge.

<question>{{input}}</question>
<document>{{doc}}</document>

Please indicate whether the document contains information that is relevant to the question using only the following json format. Do not use any markdown formatting or output additional lines.
{
  "rationale": "Reason for the assessment. If the document does not provide any information that is relevant to the question then state which parts are not relevant. Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}
`result` must only be `yes` or `no`."""  # noqa: E501


def get_prompt(request: str, context: str) -> str:
    return format_prompt(
        RETRIEVAL_RELEVANCE_PROMPT,
        input=request,
        doc=context,
    )
