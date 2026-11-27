import re
from typing import Any


def format_prompt(prompt: str, **values: Any) -> str:
    """Format double-curly variables in the prompt template."""
    if not values:
        return prompt
    # Replace all variables in a single pass, so text inserted for one variable is never treated
    # as a placeholder for another (e.g. a value that contains "{{ context }}").
    pattern = re.compile(r"\{\{\s*(" + "|".join(map(re.escape, values)) + r")\s*\}\}")
    return pattern.sub(lambda match: str(values[match.group(1)]), prompt)
