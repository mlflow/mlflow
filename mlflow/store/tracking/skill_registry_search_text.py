"""Search-text projection builder for the skill registry.

Each skill stores a ``search_text`` column that
concatenates several discovery fields into one value.  A single
``LIKE``/``ILIKE`` on this column matches across all discovery fields without
requiring ``OR`` (which MLflow ``filter_string`` does not support).

This helper *builds* the projection string.  It does **not** persist it —
callers in the concrete search stories handle persistence.
"""

from __future__ import annotations

import re

_WHITESPACE_RUN = re.compile(r"\s+")


def _normalize(text: str | None) -> str:
    if text is None:
        return ""
    # str() guards against non-string values.
    s = str(text)
    return _WHITESPACE_RUN.sub(" ", s).strip() if s else ""


def build_skill_search_text(
    name: str,
    description: str | None = None,
) -> str:
    """Build the ``search_text`` projection for a skill from discovery metadata."""
    parts = [_normalize(name), _normalize(description)]
    return " ".join(part for part in parts if part)
