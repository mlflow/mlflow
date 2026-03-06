from __future__ import annotations

# Fetch N * sample_size traces so random sampling has enough diversity
SAMPLE_POOL_MULTIPLIER = 5
SAMPLE_RANDOM_SEED = 42
# LLM calls are non-deterministic; retry on transient API/rate-limit errors
NUM_RETRIES = 5
# Upper bound on generated tokens per LLM call (covers long cluster summaries)
LLM_MAX_TOKENS = 8192

# Text truncation limits
SURFACE_TRUNCATION_LIMIT = 800


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
