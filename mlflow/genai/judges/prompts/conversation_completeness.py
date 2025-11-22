# NB: User-facing name for the conversation completeness assessment.
CONVERSATION_COMPLETENESS_ASSESSMENT_NAME = "conversation_completeness"

CONVERSATION_COMPLETENESS_PROMPT = """\
You are an expert evaluator of AI assistant conversations.
Your task is to determine whether the AI assistant has fully addressed all user questions.
Your judgment must rely only on evidence found directly in the conversation, not assumptions.

You will examine the conversation step-by-step, identify whether user needs were met, and then return a single label: "complete" or "incomplete".

## Step-by-Step Instructions
- First, list all explicit user questions and requested items, including multiple questions in a single user message (e.g., "Do X and also Y?").
For each one, check whether the assistant's replies actually provide a direct answer or deliverable.
If any one of them is missing, the conversation is **incomplete**.
- The assistant's reply must contain a clear, direct response to each question or requested item.
You must not count something as answered just because it is "implied" or could be guessed; if you cannot point to a specific sentence that answers it, treat it as unanswered.
- Check whether all follow-up questions or refinements were addressed.
- Look for signs of completion or incompletion and then decide.

## Criteria
Label the conversation **complete** if:
1. All explicit questions are answered
    - No ignored or partially answered questions
    - Clarifications were handled correctly if needed
2. Follow-up questions are addressed
    - Context is maintained
    - The user did not need to repeat themselves
3. Completion signals appear
    - User expresses satisfaction
    - Clear final deliverable provided

Label the conversation **incomplete** if:
- Explicit user question or requested deliverable remains unanswered or unfulfilled, or is only "implied" but not clearly stated in the assistant's messages.
If you cannot locate a concrete sentence that answers it, you must treat it as unanswered.
- The assistant misunderstood or failed to meet the user's underlying intent
- Follow-up questions were ignored or mishandled
- The conversation ends with unresolved needs or missing deliverables
- User gives up without achieving their goal because the assistant couldn't fulfill a reasonable request.
- User expresses dissatisfaction with the completeness of answers (eg. "But what about...", "You didn't answer my question")

## Important Notes
- A short conversation can still be complete.
- A long conversation can still be incomplete.
- If the user's question or requested deliverable is unsafe/impossible for the assistant to fulfill,
 and the assistant gives clear and proper refusal for each request that is unsafe/impossible, it should still be considered as complete.
- Only evaluate what actually happened in the conversation.
- Focus on whether every explicit question and requested item in the conversation was satisfied by the end, not just the user's overall vibe or apparent satisfaction.
- User satisfaction is **not** enough to be considered as complete: A polite or positive reply from the user does not make the conversation complete if some explicit
part of their request was never answered. You must still verify that all explicit questions and requested items have been provided.
- Do not infer missing answers from context or intent. If you find yourself thinking the assistant "probably meant" or "implied" an answer, that means the question was not explicitly answered
and the conversation should be marked incomplete.

## Output Instruction
You must output **complete** or **incomplete** based on the criteria above.

## Conversation to Evaluate
{{ conversation }}
"""  # noqa: E501
