DEFAULT_PERSONA = "You are an inquisitive user having a natural conversation."

INITIAL_USER_PROMPT = """{persona}

Your goal in this conversation is to: {goal}

Start a conversation about this topic. The way you start the conversation should sound as much like
a realistic user as possible. Don't ask about your goal directly - instead start with a broader
question and let the conversation develop naturally."""

# NB: We embed history into the prompt instead of passing a message list directly to reduce
#     noise, since the prompt only cares about message content and sender role.
FOLLOWUP_USER_PROMPT = """{persona}

Your goal: {goal}

Conversation so far:
{conversation_history}

The agent you are talking to just said: {last_response}

Respond naturally with a follow-up and guide the conversation toward your goal as a
realistic user."""

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
