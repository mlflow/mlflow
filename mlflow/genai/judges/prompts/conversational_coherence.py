# NB: User-facing name for the conversational coherence assessment.
CONVERSATIONAL_COHERENCE_ASSESSMENT_NAME = "conversational_coherence"

CONVERSATIONAL_COHERENCE_PROMPT = """\
Consider the following conversation history between a user and an assistant. Your task is to \
evaluate the logical flow and consistency of the assistant's responses across the entire \
conversation and output exactly one label: "yes" or "no".

Evaluate the assistant's responses for the following coherence criteria:
- Logical progression: Each assistant response follows logically from prior turns and preserves the \
thread of discussion rather than abruptly switching topics without reason.
- Internal consistency: The assistant does not contradict its own earlier statements, facts, or \
commitments within the conversation.
- Reference resolution: The assistant correctly resolves pronouns, ellipses, and references to \
previously discussed entities instead of losing track of subjects.
- Context retention: The assistant remembers and uses information the user has already provided, \
rather than re-asking for the same details or ignoring established context.
- Topic continuity: When the user continues a topic, the assistant stays on topic; when the user \
legitimately changes topics, the assistant transitions cleanly.

Evaluation guidelines:
- Focus exclusively on the assistant's responses. Incoherent or contradictory user messages do not \
by themselves make the conversation incoherent.
- A conversation is coherent if the assistant maintains a clear, consistent, and contextually \
appropriate dialogue across turns, even if individual turns are short.
- Asking the user to clarify ambiguous requests is coherent behavior and should not be penalized.
- Minor stylistic variation or rephrasing between turns is acceptable as long as meaning is preserved.
- Do not penalize the assistant for correcting itself when it explicitly acknowledges the correction.

Output "yes" if the assistant's responses are coherent throughout the conversation. Output "no" \
only if at least one assistant response contains a clear coherence failure as defined above.

<conversation>{{ conversation }}</conversation>
"""  # noqa: E501
