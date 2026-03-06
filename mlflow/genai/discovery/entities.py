from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _ConversationAnalysis:
    """Per-session analysis built from triage rationales and span errors.

    Args:
        surface: Truncated combined rationale shown to the LLM for labeling.
        root_cause: Full untruncated combined rationale for summarization.
        affected_trace_ids: Trace IDs of failing traces in this session.
        execution_path: Compact path of sub-agents/tools called (e.g.
            ``"ask_sports > get_scores, web_search"``).
    """

    surface: str
    root_cause: str
    affected_trace_ids: list[str]
    execution_path: str = ""
