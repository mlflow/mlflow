from __future__ import annotations

import json
import logging

from pydantic import BaseModel as _BaseModel

from mlflow.entities.issue import IssueSeverity
from mlflow.genai.discovery.constants import (
    CLUSTER_LABELS_PROMPT_TEMPLATE,
    build_cluster_summary_prompt,
)
from mlflow.genai.discovery.entities import (
    _ConversationAnalysis,
    _IdentifiedIssue,
)
from mlflow.genai.discovery.utils import _call_llm, _TokenCounter

_logger = logging.getLogger(__name__)


class _ClusterGroup(_BaseModel):
    name: str
    indices: list[int]


class _ClusterResponse(_BaseModel):
    groups: list[_ClusterGroup]


def cluster_by_llm(
    labels: list[str],
    max_issues: int,
    model: str,
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
    numbered = "\n".join(f"[{i}] {lbl}" for i, lbl in enumerate(labels))
    prompt = CLUSTER_LABELS_PROMPT_TEMPLATE.format(
        num_labels=len(labels),
        max_issues=max_issues,
        numbered_labels=numbered,
    )

    response = _call_llm(
        model,
        [{"role": "user", "content": prompt}],
        response_format=_ClusterResponse,
        token_counter=token_counter,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        _logger.debug(
            "LLM returned empty content for label grouping "
            "(finish_reason=%s), falling back to singletons",
            response.choices[0].finish_reason,
        )
        return [[i] for i in range(len(labels))]
    result = _ClusterResponse(**json.loads(content))

    # Validate indices and collect groups; orphaned indices become singletons
    clustered_indices: set[int] = set()
    cluster_groups: list[list[int]] = []
    for group in result.groups:
        if indices := [i for i in group.indices if 0 <= i < len(labels)]:
            cluster_groups.append(indices)
            clustered_indices.update(indices)

    cluster_groups.extend([i] for i in range(len(labels)) if i not in clustered_indices)

    # Enforce max_issues limit by keeping the largest groups
    if len(cluster_groups) > max_issues:
        cluster_groups.sort(key=len, reverse=True)
        cluster_groups = cluster_groups[:max_issues]

    return cluster_groups


def summarize_cluster(
    cluster_label_indices: list[int],
    analyses: list[_ConversationAnalysis],
    model: str,
    label_to_analysis: list[int] | None = None,
    categories: list[str] | None = None,
    token_counter: _TokenCounter | None = None,
) -> _IdentifiedIssue:
    """
    Summarize a cluster of analyses into a single identified issue.

    Uses an LLM to synthesize a name, description, root cause, and severity
    for the cluster. Always returns all corresponding analysis indices as
    example_indices (overriding the LLM's selection).

    Args:
        cluster_label_indices: Label indices that form this cluster.
        analyses: All conversation analyses from the pipeline.
        model: Model URI for the summarization LLM.
        label_to_analysis: Mapping from label index to analysis index.
            When ``None``, label indices are used as analysis indices directly.
        categories: Optional list of valid category names. If provided,
            extracted categories will be filtered to only include these.
        token_counter: Optional token counter for tracking LLM usage.

    Returns:
        An ``_IdentifiedIssue`` with synthesized fields and analysis indices.
    """
    # Map label indices to unique analysis indices (preserving order)
    if label_to_analysis is not None:
        analysis_indices = list(dict.fromkeys(label_to_analysis[i] for i in cluster_label_indices))
    else:
        analysis_indices = cluster_label_indices

    cluster_analyses = [analyses[i] for i in analysis_indices]
    parts = []
    for i, analysis in zip(analysis_indices, cluster_analyses):
        entry = f"[{i}] {analysis.rationale_summary}"
        if analysis.execution_path:
            entry += f"\n  execution_path: {analysis.execution_path}"
        parts.append(entry)
    analyses_text = "\n\n".join(parts)

    system_prompt = build_cluster_summary_prompt(categories=categories)
    response = _call_llm(
        model,
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Cluster of {len(analysis_indices)} analyses:\n\n{analyses_text}",
            },
        ],
        response_format=_IdentifiedIssue,
        token_counter=token_counter,
    )

    content = (response.choices[0].message.content or "").strip()
    result = _IdentifiedIssue(**json.loads(content))
    result.example_indices = analysis_indices

    # Filter categories to only include valid ones if a list was provided
    if categories is not None and result.categories:
        valid_categories = set(categories)
        result.categories = [cat for cat in result.categories if cat in valid_categories]

    return result


def recluster_singletons(
    singletons: list[_IdentifiedIssue],
    analysis_labels: dict[int, str],
    analyses: list[_ConversationAnalysis],
    model: str,
    max_issues: int,
    categories: list[str] | None = None,
    token_counter: _TokenCounter | None = None,
) -> list[_IdentifiedIssue]:
    """
    Re-cluster singleton issues via a second LLM pass to find better groupings.

    Args:
        singletons: Single-analysis issues to attempt merging.
        analysis_labels: Mapping from analysis index to its first label string.
        analyses: All conversation analyses from the pipeline.
        model: Model URI for clustering and summarization.
        max_issues: Maximum number of groups to produce.
        categories: Optional list of valid category names to filter extracted categories.
        token_counter: Optional token counter for tracking LLM usage.

    Returns:
        List of issues after re-clustering (merged or original singletons).
    """
    if len(singletons) < 2:
        return list(singletons)

    singleton_labels = []
    for singleton in singletons:
        idx = singleton.example_indices[0]
        singleton_labels.append(analysis_labels.get(idx, singleton.name))

    new_groups = cluster_by_llm(singleton_labels, max_issues, model, token_counter=token_counter)

    result: list[_IdentifiedIssue] = []
    for group in new_groups:
        if len(group) == 1:
            result.append(singletons[group[0]])
            continue
        # Each singleton has exactly one analysis index in example_indices
        merged_indices = [singletons[group_idx].example_indices[0] for group_idx in group]
        merged_issue = summarize_cluster(
            merged_indices, analyses, model, categories=categories, token_counter=token_counter
        )
        if merged_issue.severity >= IssueSeverity.LOW:
            result.append(merged_issue)
        else:
            result.extend(singletons[group_idx] for group_idx in group)

    return result
