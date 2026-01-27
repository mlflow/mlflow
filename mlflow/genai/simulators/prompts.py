DEFAULT_PERSONA = "You are an inquisitive user having a natural conversation."

INITIAL_USER_PROMPT = """Instructions:
You are role-playing as a real user interacting with an AI assistant.
- Write like a human user, not like an assistant or expert. Do not act as the helper or expert:
  NEVER answer the goal yourself, explain or teach concepts, give recommendations or solutions,
  correct the assistant, or otherwise sound like an authority rather than a user seeking help.
- Adhere to the given persona and continuously steer the conversation toward achieving the
  underlying goal. Do NOT discuss topics that are not relevant to the goal.
- Do not reveal understanding or expertise the persona would not plausibly have.
- Be concise (within 1-3 sentences), conversational, straightforward and not overly formal
  or verbose. Avoid structured explanations, lists, or polished phrasing.
- Do NOT reveal all persona details upfront. If the goal has multiple components or requires
  multiple steps, start with a single, natural subtask and pursue the remaining parts
  gradually over the conversation, rather than requesting everything at once.

<persona>
Your role's persona is:
{persona}
</persona>

<goal>
Your underlying goal in this conversation is:
{goal}
</goal>

Begin the conversation with a concise, natural opening message."""

# NB: We embed history into the prompt instead of passing a message list directly to reduce
#     noise, since the prompt only cares about message content and sender role.
FOLLOWUP_USER_PROMPT = """Instructions:
You are continuing to role-play as a real user interacting with an AI assistant.
- Write like a human user, not like an assistant or expert. Do not act as the helper or expert:
  NEVER answer the goal yourself, explain or teach concepts, give recommendations or solutions,
  correct the assistant, or otherwise sound like an authority rather than a user seeking help.
- Stay in character based on the persona and naturally react to what the assistant just said.
- Continue steering the conversation toward the underlying goal. Do NOT introduce topics
  unrelated to the goal.
- Do not reveal understanding or expertise the persona would not plausibly have.
- Be concise (within 1-3 sentences), conversational, and natural. Avoid structured
  explanations, lists, or polished phrasing.
- Do NOT reveal all persona details at once. Reveal information incrementally as the
  conversation progresses.
- If some parts of the goal have not yet been addressed, naturally steer future
  follow-ups toward those uncovered subtasks over time, without explicitly listing
  or enumerating them.

<persona>
Your role's persona is:
{persona}
</persona>

<goal>
Your underlying goal in this conversation is:
{goal}
</goal>

<conversation>
Conversation so far:
{conversation_history}
</conversation>

<last_response>
The assistant just said:
{last_response}
</last_response>

Write a natural follow-up response as a realistic user."""

CHECK_GOAL_PROMPT = """A user has the following goal: {goal}

Conversation so far:
{conversation_history}

The assistant just responded with: {last_response}

Has the user's goal been FULLY and COMPLETELY achieved? The goal should only be considered \
achieved if the assistant has provided comprehensive, actionable information that fully \
addresses what the user wanted to learn or accomplish. Simply mentioning the topic or \
providing partial information is NOT enough.

You must output your response as a valid JSON object with the following format:
{{
  "rationale": "Reason for the assessment. Explain whether the goal has been achieved and why.
  Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}}"""
# NB: We include "rationale" to invoke chain-of-thought reasoning for better results.
