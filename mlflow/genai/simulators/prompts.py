DEFAULT_PERSONA = "You are a helpful user having a natural conversation."

INITIAL_USER_PROMPT = """{persona}

Your goal in this conversation is to: {goal}

Start a conversation about this topic. Don't ask about your goal directly - \
instead start with a broader question and let the conversation develop naturally."""

FOLLOWUP_USER_PROMPT = """{persona}

Your goal: {goal}

Conversation so far:
{conversation_history}

They just said: {last_response}

Respond naturally and guide the conversation toward your goal."""

CHECK_GOAL_PROMPT = """Goal: {goal}

Latest response: {last_response}

Has the conversation achieved the specified goal? Answer only 'yes' or 'no'."""
