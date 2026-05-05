"""LLM-judge prompts and the wrappers that call them.

Two judges:

- `cluster_judge`: given two lists of findings (e.g. reviewer-standalone and
  reviewer-opinion), assigns each finding to a cluster of equivalent concerns.
- `semantic_dedup_judge`: given a draft finding and the bodies of nearby open
  human threads, decides whether the finding describes the same concern as
  any of them.
"""

from __future__ import annotations

import json
from typing import Literal

from orchestrator.anthropic_client import AnthropicClient, Message, ModelChoice
from orchestrator.cluster import SourceFinding

_CLUSTER_JUDGE_SYSTEM = """You are a semantic-overlap judge for a code-review eval.

You receive two lists of code-review findings on the same PR from two
different review architectures. Your job is to cluster them by semantic
equivalence: two findings cluster together if they describe the SAME
concern (same root cause / same suggested fix), even if wording differs.
Path+line proximity is a strong signal but not required.

Rules:
- Every finding from every list must appear in EXACTLY ONE cluster
  (singletons are clusters with only one source).
- Be conservative on matching: when in doubt, keep separate. Precision
  matters more than aggressive merging.
- Do not merge a "missing test" finding with a "logic bug" finding even
  if they touch the same line.

Output ONLY a JSON object of this shape:

{
  "clusters": [
    {"a": [<index>, ...], "b": [<index>, ...]},
    ...
  ]
}

Where `a` indices reference list A, `b` indices reference list B. Each list
of indices in a cluster may be empty (singleton from the other list).
"""


def _format_finding_list(findings: list[SourceFinding]) -> str:
    out = []
    for i, f in enumerate(findings):
        out.append(f"[{i}] {f.path}:{f.line} (severity={f.severity}): {f.body}")
    return "\n".join(out)


async def cluster_judge(
    client: AnthropicClient,
    list_a: list[SourceFinding],
    list_b: list[SourceFinding],
    *,
    models: ModelChoice,
) -> str:
    """Return the raw judge output (JSON string) for materialize_clusters()."""
    user_content = (
        "List A:\n"
        f"{_format_finding_list(list_a) or '(empty)'}\n\n"
        "List B:\n"
        f"{_format_finding_list(list_b) or '(empty)'}\n"
    )
    result = await client.complete(
        role=models.cluster_judge,
        system=_CLUSTER_JUDGE_SYSTEM,
        messages=[Message(role="user", content=user_content)],
        max_tokens=4096,
    )
    return result.text


_SEMANTIC_DEDUP_SYSTEM = """You decide whether a draft code-review finding
describes the same concern as any of a set of existing open review threads.

Two findings are the same concern if they share the root cause or the
suggested fix. Two findings on the same line about different aspects (one
about a logic bug, one about a missing test) are NOT the same concern.

Be conservative. If the draft finding adds value over the existing threads,
return false so the bot posts. If the draft would be a "+1" or "ditto" of
an existing thread, return true so the bot stays silent.

Output ONLY a JSON object: {"same_concern": true} or {"same_concern": false}.
"""


async def semantic_dedup_judge(
    client: AnthropicClient,
    finding_body: str,
    existing_thread_bodies: list[str],
    *,
    models: ModelChoice,
) -> bool:
    if not existing_thread_bodies:
        return False
    threads_block = "\n\n".join(
        f"--- thread {i + 1} ---\n{body}" for i, body in enumerate(existing_thread_bodies)
    )
    user_content = f"Draft finding:\n{finding_body}\n\nExisting open threads:\n{threads_block}"
    result = await client.complete(
        role=models.semantic_dedup_judge,
        system=_SEMANTIC_DEDUP_SYSTEM,
        messages=[Message(role="user", content=user_content)],
        max_tokens=128,
    )
    return _parse_same_concern(result.text)


def _parse_same_concern(text: str) -> bool:
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return False
    return bool(data.get("same_concern"))


def make_semantic_judge(client: AnthropicClient, models: ModelChoice):
    """Adapter to satisfy the `dedup.SemanticJudge` Protocol shape."""

    async def _judge(finding_body: str, thread_bodies: list[str]) -> bool:
        return await semantic_dedup_judge(client, finding_body, thread_bodies, models=models)

    return _judge


# Reviewer agent invocation prompts (as user-message content; the system
# prompt is the loaded agent .md body).


def reviewer_discovery_user_message(
    pr_number: int,
    pr_title: str,
    diff: str,
    file_contents: dict[str, str],
) -> str:
    file_block = "\n\n".join(
        f"=== {path} (full file at PR head) ===\n{content}"
        for path, content in file_contents.items()
    )
    return (
        f"PR #{pr_number}: {pr_title}\n\n"
        f"=== diff ===\n{diff}\n\n"
        f"{file_block}\n\n"
        "You are in DISCOVERY mode. Output ONLY the JSON specified in your "
        "Output format (Discovery mode) section."
    )


def reviewer_opinion_user_message(
    pr_number: int,
    pr_title: str,
    diff: str,
    spotter_findings_json: str,
) -> str:
    return (
        f"PR #{pr_number}: {pr_title}\n\n"
        f"=== diff ===\n{diff}\n\n"
        f"=== spotter_findings ===\n{spotter_findings_json}\n\n"
        "You are in OPINION mode. Output ONLY the JSON specified in your "
        "Output format (Opinion mode) section."
    )


def spotter_user_message(
    pr_number: int,
    pr_title: str,
    diff: str,
    file_contents: dict[str, str],
) -> str:
    file_block = "\n\n".join(
        f"=== {path} (full file at PR head) ===\n{content}"
        for path, content in file_contents.items()
    )
    return (
        f"PR #{pr_number}: {pr_title}\n\n"
        f"=== diff ===\n{diff}\n\n"
        f"{file_block}\n\n"
        "Output ONLY the JSON specified in your Output format section. Do "
        "not consider existing PR comments."
    )


# Re-exports for type checking
_AssistantParse = Literal["reviewer", "spotter"]
