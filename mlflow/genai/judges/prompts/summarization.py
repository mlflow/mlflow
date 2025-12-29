# NB: User-facing name for the summarization assessment.
SUMMARIZATION_ASSESSMENT_NAME = "summarization"

SUMMARIZATION_PROMPT = """\
Consider the following source document and candidate summary.
You must decide whether the summary is an acceptable summary of the document.
Output only "yes" or "no" based on whether the summary meets the criteria below.

First, read the document and summary carefully.
Second, evaluate faithfulness: check whether every concrete claim in the summary is supported by the document. Emphasize the accuracy of the main facts rather than the exact phrasing. If the summary contradicts the document or invents information, it fails.
Third, evaluate coverage: identify the main points of the document and determine whether the summary captures all of the important ideas. It may omit minor details, examples, and repetitions, but it should not miss any major point or distort their relative importance.
Fourth, evaluate conciseness and focus: the summary must substantially compress the document into its essential ideas. It is not sufficient for the summary to merely be shorter than the original. Overly long summaries that closely paraphrase large portions of the document fail.
Fifth, evaluate clarity and coherence: the summary should be understandable, logically organized, and free of serious grammatical or structural issues that make its meaning unclear. Minor language errors are acceptable if they do not interfere with understanding.

Return "yes" only if all of the following are true:
The summary is faithful to the document (no hallucinations or contradictions).
The summary covers all major ideas in the document without omitting important points.
The summary is concise and focused while still preserving those major ideas.
The summary is clear enough to be easily understood.

If any of these conditions are not satisfied, return "no".

<document>{{ inputs }}</document>

<summary>{{ outputs }}</summary>
"""  # noqa: E501
