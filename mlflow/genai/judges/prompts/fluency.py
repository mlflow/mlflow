# NB: User-facing name for the fluency assessment.
FLUENCY_ASSESSMENT_NAME = "fluency"

FLUENCY_PROMPT = """\
You are a linguistic expert evaluating the Fluency of AI-generated text in {{ outputs }}.

Definition: Fluency measures the grammatical correctness, natural flow, and linguistic quality
of the text, regardless of factual accuracy.

Evaluation Checklist:
- Grammar: Is the text free of spelling and grammatical errors?
- Naturalness: Does it read like natural human writing, avoiding "stiff" or "robotic" phrasing?
- Flow: Do sentences transition smoothly, or is the text choppy?
- Variety: Is there variation in sentence structure and vocabulary?
"""
