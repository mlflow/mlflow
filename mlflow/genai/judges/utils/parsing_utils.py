"""Response parsing utilities for judge models."""

import re


def _strip_markdown_code_blocks(response: str) -> str:
    """
    Strip markdown code blocks from LLM responses.

    Some legacy models wrap responses in markdown code blocks (```json...``` or
    unlabeled fences). This function removes those wrappers to extract the raw content.
    It also handles cases where a ```json block is embedded within a larger response
    that contains preamble text before the code block.
    Fences on the same line as the content (e.g. ```json {...}```) are stripped too.

    Args:
        response: The raw response from the LLM

    Returns:
        The response with markdown code blocks removed
    """
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if len(lines) == 1:
            # Single-line block, e.g. ```json {"result": "yes"}```
            return re.sub(r"^```[\w-]*|```$", "", cleaned).strip()

        start_idx = 1
        end_idx = len(lines)
        found_closing_fence = False

        for i, line in enumerate(lines):
            if i == 0 and line.startswith("```"):
                start_idx = 1
            elif line.strip() == "```" and i > 0:
                end_idx = i
                found_closing_fence = True
                break

        if not found_closing_fence:
            # No closing fence on its own line, so it may be glued to the end of the content.
            # Only checked as a fallback: with json.loads(strict=False), a line inside a
            # multi-line JSON string can also end with ```.
            for i, line in enumerate(lines[1:], 1):
                if line.rstrip().endswith("```"):
                    lines[i] = line.rstrip().removesuffix("```")
                    end_idx = i + 1
                    break

        return "\n".join(lines[start_idx:end_idx]).strip()

    # Handle embedded code blocks (preamble text followed by a ```json code block)
    if json_block_match := re.search(r"```json\s*\n(.*?)\n```", cleaned, re.DOTALL | re.IGNORECASE):
        return json_block_match.group(1).strip()

    return cleaned


def _sanitize_justification(justification: str) -> str:
    return justification.replace("Let's think step by step. ", "")
