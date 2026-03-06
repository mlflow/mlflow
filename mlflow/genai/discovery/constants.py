from __future__ import annotations

from typing import Literal

# Fetch N * sample_size traces so random sampling has enough diversity
SAMPLE_POOL_MULTIPLIER = 5
SAMPLE_RANDOM_SEED = 42
# LLM calls are non-deterministic; retry on transient API/rate-limit errors
NUM_RETRIES = 5
# Upper bound on generated tokens per LLM call (covers long cluster summaries)
LLM_MAX_TOKENS = 8192

# Text truncation limits
SURFACE_TRUNCATION_LIMIT = 800

# Severity scale — ordinal comparison for filtering/sorting
SeverityLevel = Literal["not_an_issue", "low", "medium", "high"]
SEVERITY_LEVELS = ("not_an_issue", "low", "medium", "high")
SEVERITY_ORDER = {level: i for i, level in enumerate(SEVERITY_LEVELS)}
MIN_SEVERITY = "low"

DEFAULT_MODEL = "openai:/gpt-5-mini"


# ---- Failure label extraction prompt ----

FAILURE_LABEL_SYSTEM_PROMPT = (
    "You extract a short failure symptom from a conversation analysis. "
    "Describe WHAT WENT WRONG from the user's perspective in 5-15 words.\n\n"
    "Briefly mention the domain or topic so the label has context, "
    "but keep the focus on the observable symptom.\n\n"
    "Examples:\n"
    '- "didn\'t provide current S&P 500 futures despite explicit request"\n'
    '- "failed to resume Spotify playback despite repeated user requests"\n'
    '- "gave wrong lyric count, did not correct when challenged"\n'
    '- "omitted requested sources and citations for news query"\n'
    '- "contradicted itself on cheesecake shelf-life across responses"\n'
    '- "ignored user\'s stop command and continued suggesting topics"\n\n'
    "Return ONLY the symptom, nothing else."
)

# ---- Cluster summary prompt ----

NO_ISSUE_KEYWORD = "NO_ISSUE_DETECTED"

CLUSTER_SUMMARY_SYSTEM_PROMPT = (
    "You are an expert at analyzing AI application failures. You will be given a group of "
    "per-conversation failure analyses that were pre-clustered by semantic similarity.\n\n"
    "Your job is to:\n"
    "1. **Summarize** the cluster into a single issue with a name, description, and root cause\n"
    "2. **Validate** whether the grouped analyses actually represent the same underlying issue\n\n"
    "IMPORTANT: If the analyses do NOT represent a real failure — e.g. the user's goals were "
    "achieved, the system functioned correctly, or there is no concrete deficiency — you MUST "
    f'set the name to exactly "{NO_ISSUE_KEYWORD}" and set severity to "not_an_issue". '
    "Do NOT invent an issue where none exists.\n\n"
    "Provide:\n"
    "- A name prefixed with 'Issue: ' followed by a short readable description "
    "(3-8 words, plain English), e.g. 'Issue: Media control commands ignored', "
    f"'Issue: Incorrect data returned' — or exactly \"{NO_ISSUE_KEYWORD}\" if no real issue\n"
    "- A description of what specifically went wrong from the user's perspective. "
    "Cite observable symptoms (e.g. 'returned empty response', 'ignored the user's "
    "constraint to avoid implementation'). Avoid vague language like 'inefficient' "
    "or 'suboptimal' without concrete details.\n"
    "- The root cause: why this likely happens AND where to investigate. Identify "
    "the probable component, behavior, or configuration at fault (e.g. 'the retrieval "
    "tool may be returning stale cached results', 'the system prompt does not instruct "
    "the agent to respect user constraints'). Be specific but note these are hypotheses "
    "based on observed symptoms.\n"
    f"- A severity level from: {', '.join(SEVERITY_LEVELS)}. "
    "Use medium or high only if the analyses clearly share the same failure "
    "pattern. Use not_an_issue if they do NOT belong together or represent no real issue."
)

# ---- Label clustering prompt ----

CLUSTER_LABELS_PROMPT_TEMPLATE = (
    "Below are {num_labels} failure labels from an AI agent.\n"
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
    "- Create at most {max_issues} groups\n\n"
    "Labels:\n{numbered_labels}\n\n"
    'Return a JSON object with a "groups" key containing an array of objects, '
    'each with "name" (short readable string) and "indices" (list of ints).\n'
    "Return ONLY the JSON, no explanation."
)
