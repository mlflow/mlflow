from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import mlflow
from mlflow.genai.discovery.constants import (
    CLUSTER_SUMMARY_SYSTEM_PROMPT,
    DEFAULT_JUDGE_MODEL,
    LLM_MAX_TOKENS,
    NUM_RETRIES,
)
from mlflow.genai.discovery.entities import (
    _ConversationAnalysis,
    _IdentifiedIssue,
)
from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm
from mlflow.metrics.genai.model_utils import convert_model_uri_to_litellm

if TYPE_CHECKING:
    from mlflow.genai.discovery.entities import Issue
    from mlflow.genai.discovery.pipeline import _TokenCounter

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
    model = model or DEFAULT_JUDGE_MODEL
    litellm_model = convert_model_uri_to_litellm(model)

    numbered = "\n".join(f"[{i}] {lbl}" for i, lbl in enumerate(labels))
    prompt = (
        f"Below are {len(labels)} failure labels from an AI agent.\n"
        "Each label has the format: [execution_path] symptom\n"
        "The execution path shows which sub-agents and tools were called.\n\n"
        "Group these labels into coherent issue categories. Two labels belong "
        "in the same group when:\n"
        "  1. They share the same failure pattern (similar symptom)\n"
        "  2. They involve the same tool, sub-agent, or execution path\n\n"
        "Same tool/path strongly suggests the same root cause — group together "
        "unless symptoms are clearly unrelated. Different paths MAY still be the "
        "same issue if symptoms are very similar.\n\n"
        "Rules:\n"
        "- Each group should have a name prefixed with 'Issue: ' followed by a short "
        "readable description (3-8 words), e.g. 'Issue: Incomplete response details'\n"
        "- A label can only appear in one group\n"
        "- Singleton groups are fine for truly unique issues\n"
        f"- Create at most {max_issues} groups\n\n"
        f"Labels:\n{numbered}\n\n"
        'Return a JSON object with a "groups" key containing an array of objects, '
        'each with "name" (short readable string) and "indices" (list of ints).\n'
        "Return ONLY the JSON, no explanation."
    )

    response = _invoke_litellm(
        litellm_model=litellm_model,
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        num_retries=NUM_RETRIES,
        response_format={"type": "json_object"},
        include_response_format=True,
        inference_params={"max_tokens": LLM_MAX_TOKENS, "temperature": 0},
    )
    if token_counter is not None:
        token_counter.track(response)
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
    analysis_model: str,
    token_counter: _TokenCounter | None = None,
) -> _IdentifiedIssue:
    """
    Summarize a cluster of analyses into a single identified issue.

    Uses an LLM to synthesize a name, description, root cause, and confidence
    for the cluster. Always returns all cluster indices as example_indices
    (overriding the LLM's selection).

    Args:
        cluster_indices: Indices into ``analyses`` that form this cluster.
        analyses: All conversation analyses from the pipeline.
        analysis_model: Model URI for the summarization LLM.
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

    litellm_model = convert_model_uri_to_litellm(analysis_model)

    response = _invoke_litellm(
        litellm_model=litellm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Cluster of {len(cluster_indices)} analyses:\n\n{analyses_text}",
            },
        ],
        tools=[],
        num_retries=NUM_RETRIES,
        response_format={"type": "json_object"},
        include_response_format=True,
        inference_params={"max_tokens": LLM_MAX_TOKENS, "temperature": 0},
    )
    if token_counter is not None:
        token_counter.track(response)

    content = (response.choices[0].message.content or "").strip()
    result = _IdentifiedIssue(**json.loads(content))
    result.example_indices = cluster_indices
    return result


def build_summary(issues: list[Issue], total_traces: int) -> str:
    if not issues:
        return f"## Issue Discovery Summary\n\nAnalyzed {total_traces} traces. No issues found."

    lines = [
        "## Issue Discovery Summary\n",
        f"Analyzed **{total_traces}** traces. Found **{len(issues)}** issues:\n",
    ]
    for i, issue in enumerate(issues, 1):
        lines.append(
            f"### {i}. {issue.name} ({issue.frequency:.0%} of traces, "
            f"confidence: {issue.confidence})\n\n"
            f"{issue.description}\n\n"
            f"**Root cause:** {issue.root_cause}\n"
        )
    return "\n".join(lines)


def log_discovery_artifacts(run_id: str, artifacts: dict[str, str]) -> None:
    if not run_id:
        return
    client = mlflow.MlflowClient()
    for filename, content in artifacts.items():
        try:
            client.log_text(run_id, content, filename)
        except Exception:
            _logger.warning("Failed to log %s to run %s", filename, run_id, exc_info=True)
