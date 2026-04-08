HALLUCINATION_FEEDBACK_NAME = "hallucination_detection"

HALLUCINATION_PROMPT = """\
Determine if {{ outputs }} (the AI response) is faithful to {{ inputs }} (the reference context).

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
