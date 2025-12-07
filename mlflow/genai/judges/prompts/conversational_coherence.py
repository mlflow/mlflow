# NB: User-facing name for the conversational coherence assessment.
CONVERSATIONAL_COHERENCE_ASSESSMENT_NAME = "conversational_coherence"

CONVERSATIONAL_COHERENCE_PROMPT = """\
Consider the following conversation history between a user and an assistant. Your task is to \
evaluate the coherence of the conversation and output exactly one label: "yes" or "no".

Evaluate the conversation for the following coherence criteria:
- Logical flow: Each response should logically follow from the previous message without abrupt \
or confusing jumps.
- Information consistency: Information provided in earlier turns should remain consistent \
throughout. The assistant should not contradict itself or provide conflicting information.
- Contextual awareness: The assistant should demonstrate understanding of and build upon the \
context established in previous turns. It should not ignore or forget relevant information.
- Topic continuity: When topics change, transitions should be smooth and understandable. Random \
or jarring topic shifts without acknowledgment indicate poor coherence.
- Relevance maintenance: All responses should be relevant to the user's questions or the ongoing \
discussion.

Evaluation guidelines:
- Focus on whether the conversation flows naturally and makes logical sense as a whole.
- Minor topic changes are acceptable if handled smoothly.
- The assistant correctly building upon context from earlier turns demonstrates good coherence.
- Contradicting previous statements or ignoring established context indicates incoherence.

Output "yes" if the conversation is coherent. Output "no" if there are clear coherence issues \
such as contradictions, context being ignored, or illogical jumps.

<conversation>{{ conversation }}</conversation>
"""  # noqa: E501
