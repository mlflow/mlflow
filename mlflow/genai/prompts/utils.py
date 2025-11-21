import re
from typing import Any

# Default TTL for prompt cache (60 seconds, matching LangFuse default)
DEFAULT_PROMPT_CACHE_TTL_SECONDS = 60


def format_prompt(prompt: str, **values: Any) -> str:
    """Format double-curly variables in the prompt template."""
    for key, value in values.items():
        # Escape backslashes in the replacement string to prevent re.sub from interpreting
        # them as escape sequences (e.g. \u being treated as Unicode escape)
        replacement = str(value).replace("\\", "\\\\")
        prompt = re.sub(r"\{\{\s*" + key + r"\s*\}\}", replacement, prompt)
    return prompt
