from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _ConversationAnalysis:
    surface: str
    root_cause: str
    affected_trace_ids: list[str]
    execution_path: str = ""
