from __future__ import annotations

import json
import logging

from mlflow.genai.discovery.constants import (
    CLUSTER_LABELS_PROMPT_TEMPLATE,
    CLUSTER_SUMMARY_SYSTEM_PROMPT,
    DEFAULT_MODEL,
    MIN_SEVERITY,
    severity_gte,
)
from mlflow.genai.discovery.entities import (
    _ConversationAnalysis,
    _IdentifiedIssue,
)
from mlflow.genai.discovery.utils import _call_llm, _TokenCounter

_logger = logging.getLogger(__name__)


def cluster_by_llm(
    labels: list[str],
    max_issues: int,
    model: str | None = None,
    token_counter: _TokenCounter | None = None,
) -> list[list[int]]:
    """
    Use an LLM to group failure labels by execution path and symptom.

    Each label has the format ``[execution_path] symptom``, where the
    execution path shows which sub-agents/tools were called. The LLM
    groups labels that share similar execution paths AND similar failure
    symptoms into coherent issue categories.

    Args:
        labels: Failure labels to cluster.
        max_issues: Maximum number of groups to produce.
        model: Model URI for the clustering LLM.
        token_counter: Optional token counter for tracking LLM usage.

    Returns:
        List of index lists, where each inner list is a cluster of label indices.
    """
    model = model or DEFAULT_MODEL

    numbered = "\n".join(f"[{i}] {lbl}" for i, lbl in enumerate(labels))
    prompt = CLUSTER_LABELS_PROMPT_TEMPLATE.format(
        num_labels=len(labels),
        max_issues=max_issues,
        numbered_labels=numbered,
    )

    response = _call_llm(
        model,
        [{"role": "user", "content": prompt}],
        json_mode=True,
        token_counter=token_counter,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        _logger.warning(
            "LLM returned empty content for label grouping "
            "(finish_reason=%s), falling back to singletons",
            response.choices[0].finish_reason,
        )
        return [[i] for i in range(len(labels))]
    result = json.loads(content)

    # Normalize response format: accept both {"groups": [...]} and bare list
    groups = (
        result if isinstance(result, list) else result.get("groups", result.get("categories", []))
    )

    # Ensure every index appears exactly once; orphaned indices become singletons
    all_indices = set()
    cluster_groups: list[list[int]] = []
    for group in groups:
        indices = [i for i in group["indices"] if 0 <= i < len(labels)]
        if indices := [i for i in indices if i not in all_indices]:
            cluster_groups.append(indices)
            all_indices.update(indices)

    cluster_groups.extend([i] for i in range(len(labels)) if i not in all_indices)

    # Enforce max_issues limit by keeping the largest groups
    if len(cluster_groups) > max_issues:
        cluster_groups.sort(key=len, reverse=True)
        cluster_groups = cluster_groups[:max_issues]

    return cluster_groups


def summarize_cluster(
    cluster_indices: list[int],
    analyses: list[_ConversationAnalysis],
    model: str,
    token_counter: _TokenCounter | None = None,
) -> _IdentifiedIssue:
    """
    Summarize a cluster of analyses into a single identified issue.

    Uses an LLM to synthesize a name, description, root cause, and severity
    for the cluster. Always returns all cluster indices as example_indices
    (overriding the LLM's selection).

    Args:
        cluster_indices: Indices into ``analyses`` that form this cluster.
        analyses: All conversation analyses from the pipeline.
        model: Model URI for the summarization LLM.
        token_counter: Optional token counter for tracking LLM usage.

    Returns:
        An ``_IdentifiedIssue`` with synthesized fields and all cluster indices.
    """
    cluster_analyses = [analyses[i] for i in cluster_indices]
    parts = []
    for i, analysis in zip(cluster_indices, cluster_analyses):
        entry = f"[{i}] {analysis.surface}"
        if analysis.execution_path:
            entry += f"\n  execution_path: {analysis.execution_path}"
        parts.append(entry)
    analyses_text = "\n\n".join(parts)

    schema_json = json.dumps(_IdentifiedIssue.model_json_schema(), indent=2)
    system_prompt = (
        f"{CLUSTER_SUMMARY_SYSTEM_PROMPT}\n\n"
        f"Respond with a JSON object matching this schema:\n{schema_json}"
    )

    response = _call_llm(
        model,
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Cluster of {len(cluster_indices)} analyses:\n\n{analyses_text}",
            },
        ],
        json_mode=True,
        token_counter=token_counter,
    )

    content = (response.choices[0].message.content or "").strip()
    result = _IdentifiedIssue(**json.loads(content))
    result.example_indices = cluster_indices
    return result


def recluster_singletons(
    singletons: list[_IdentifiedIssue],
    labels: list[str],
    analyses: list[_ConversationAnalysis],
    model: str,
    max_issues: int,
    token_counter: _TokenCounter | None = None,
) -> list[_IdentifiedIssue]:
    """
    Re-cluster singleton issues via a second LLM pass to find better groupings.

    Args:
        singletons: Single-analysis issues to attempt merging.
        labels: Failure labels from the initial clustering phase.
        analyses: All conversation analyses from the pipeline.
        model: Model URI for clustering and summarization.
        max_issues: Maximum number of groups to produce.
        token_counter: Optional token counter for tracking LLM usage.

    Returns:
        List of issues after re-clustering (merged or original singletons).
    """
    if len(singletons) < 2:
        return list(singletons)

    singleton_labels = []
    for singleton in singletons:
        idx = singleton.example_indices[0]
        singleton_labels.append(labels[idx] if idx < len(labels) else singleton.name)

    new_groups = cluster_by_llm(singleton_labels, max_issues, model, token_counter=token_counter)

    result: list[_IdentifiedIssue] = []
    for group in new_groups:
        if len(group) == 1:
            result.append(singletons[group[0]])
            continue
        merged_indices = [singletons[group_idx].example_indices[0] for group_idx in group]
        merged_issue = summarize_cluster(
            merged_indices, analyses, model, token_counter=token_counter
        )
        if severity_gte(merged_issue.severity, MIN_SEVERITY):
            result.append(merged_issue)
        else:
            result.extend(singletons[group_idx] for group_idx in group)

    return result
