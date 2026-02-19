"""Response parsing utilities for judge models."""


def _strip_markdown_code_blocks(response: str) -> str:
    """
    Strip markdown code blocks from LLM responses.

    Some legacy models wrap JSON responses in markdown code blocks (```json...```).
    This function removes those wrappers to extract the raw JSON content.

    Args:
        response: The raw response from the LLM

    Returns:
        The response with markdown code blocks removed
    """
    cleaned = response.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.split("\n")
    start_idx = 0
    end_idx = len(lines)

    for i, line in enumerate(lines):
        if i == 0 and line.startswith("```"):
            start_idx = 1
        elif line.strip() == "```" and i > 0:
            end_idx = i
            break

    return "\n".join(lines[start_idx:end_idx])


def _sanitize_justification(justification: str) -> str:
    return justification.replace("Let's think step by step. ", "")
