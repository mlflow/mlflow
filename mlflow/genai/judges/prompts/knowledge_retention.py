# NB: User-facing name for the knowledge retention assessment.
KNOWLEDGE_RETENTION_ASSESSMENT_NAME = "knowledge_retention"

KNOWLEDGE_RETENTION_PROMPT = """\
Your task is to evaluate the LAST AI response in the {{ conversation }} and determine if it:
- Correctly uses or references information the user provided in earlier turns
- Avoids contradicting information the user provided in earlier turns
- Accurately recalls details without distortion

Output "yes" if the AI's last response correctly retains any referenced prior user information.
Output "no" if the AI's last response:
- Contradicts information the user provided earlier
- Misrepresents or inaccurately recalls user-provided information
- Forgets or ignores information that is directly relevant to answering the current user question

IMPORTANT GUIDELINES:
1. Only evaluate information explicitly provided by the USER in prior turns
2. Focus on factual information (names, dates, preferences, context) rather than opinions
3. Not all prior information needs to be referenced in every response - only evaluate information
   that is directly relevant to the current user's question or request
4. If the AI doesn't reference prior information because it's not relevant to the current turn,
   that's acceptable (output "yes")
5. Only output "no" if there's a clear contradiction, distortion, or problematic forgetting of
   information that should have been used
6. Evaluate ONLY the last AI response, not the entire conversation

Base your judgment strictly on the conversation content provided. Do not use outside knowledge.
"""
