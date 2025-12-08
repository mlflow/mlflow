# A special tag in RegisteredModel to indicate that it is a prompt
IS_PROMPT_TAG_KEY = "mlflow.prompt.is_prompt"

# A special tag in ModelVersion to store the prompt text
PROMPT_TEXT_TAG_KEY = "mlflow.prompt.text"

# OSS tags (dots allowed)
PROMPT_TYPE_TAG_KEY = "mlflow.prompt.type"
RESPONSE_FORMAT_TAG_KEY = "mlflow.prompt.response_format"
LINKED_PROMPTS_TAG_KEY = "mlflow.prompt.linked_prompts"

# Unity Catalog-compatible tags (no dots)
PROMPT_TYPE_TAG_KEY_UC = "_mlflow_prompt_type"
RESPONSE_FORMAT_TAG_KEY_UC = "_mlflow_prompt_response_format"

# Prompt types
PROMPT_TYPE_TEXT = "text"
PROMPT_TYPE_CHAT = "chat"
PROMPT_TYPE_JINJA2 = "jinja2"   # ← 너가 추가해야 하는 부분 맞음

# Associated runs & experiments
PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY = "mlflow.prompt.associatedRunIds"
PROMPT_EXPERIMENT_IDS_TAG_KEY = "_mlflow_experiment_ids"

# Template variable regex
import re
PROMPT_TEMPLATE_VARIABLE_PATTERN = re.compile(
    r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\}\}"
)

PROMPT_TEXT_DISPLAY_LIMIT = 30

# Valid prompt name rule
PROMPT_NAME_RULE = re.compile(r"^[a-zA-Z0-9_.-]+$")
