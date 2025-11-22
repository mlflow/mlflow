# NB: User-facing name for the completeness assessment.
COMPLETENESS_ASSESSMENT_NAME = "completeness"

COMPLETENESS_PROMPT = """\
You are an expert evaluator of AI assistant responses.
Your task is to determine whether the AI assistant has fully addressed all user questions in a single user prompt.
Your judgment must rely only on evidence found directly in the user's question and the AI's response, not assumptions.

You will examine the interaction, identify whether the user's questions and requests were addressed, and then return a single label: "complete" or "incomplete".

## Step-by-Step Instructions
- First, identify and list all explicit user questions and requested items in the user's message.
- For each one, apply the label criteria below to determine if the response is complete or incomplete.
- Output **complete** or **incomplete** as the final answer.

## Criteria
Label the response **complete** if:
- All explicit questions and requested items are addressed.
    - No ignored or partially answered questions
    - All requested information or deliverables are provided
    - The assistant clearly refuses **and** explicitly provide the reason why it cannot answer or fulfill the request by quoting safety, legality, policy, or capability limits, etc. in its response,
    and does not ignore any other parts of the prompt it could reasonably address.
- Or the assistant explicitly requests the missing information that is genuinely necessary to answer the question or fulfill the request, and does not ignore any other parts it could already answer.

Label the response **incomplete** if:
- Any explicit user question or requested deliverable remains unanswered or unfulfilled, or is only "implied" but not clearly stated in the assistant's response.
If you cannot locate a concrete sentence that answers it, you must treat it as unanswered.
- The assistant misunderstood or failed to meet the user's underlying intent.
- Important details or information explicitly requested by the user are missing.
- The assistant simply refuses to answer and but does not explicitly provide the reason.

## Important Notes
- A short response can still be complete if it answers all questions.
- A long response can still be incomplete if it misses any explicit question.
- Do not judge whether the assistant's answer is factually correct or high-quality; only judge whether it has explicitly addressed all questions and requested items.
- You **MUST NOT** infer missing answers from context or intent. If you cannot locate a concrete sentence that explicitly answers it, you must treat it as unanswered.

## Output Instruction
You must output **complete** or **incomplete** based on the criteria above.

## User Prompt
{{ inputs }}

## AI Response
{{ outputs }}
"""  # noqa: E501
