from typing import Any

# Support unwrapped JSON with these keys for LLM use cases of Chat, Completions, Embeddings tasks
LLM_CHAT_KEY = "messages"
LLM_COMPLETIONS_KEY = "prompt"
LLM_EMBEDDINGS_KEY = "input"
SUPPORTED_LLM_FORMATS = {LLM_CHAT_KEY, LLM_COMPLETIONS_KEY, LLM_EMBEDDINGS_KEY}


def is_unified_llm_input(json_input: dict[str, Any]):
    return any(x in json_input for x in SUPPORTED_LLM_FORMATS)
