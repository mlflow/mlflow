# A special tag in RegisteredModel to indicate that it is a prompt
import re

IS_PROMPT_TAG_KEY = "mlflow.prompt.is_prompt"
# A special tag in ModelVersion to store the prompt text
PROMPT_TEXT_TAG_KEY = "mlflow.prompt.text"

# The CreateModelVersion API requires a source, but prompt versions never use it. These are the
# placeholder values MLflow's own senders use; the server rejects every other schemeless prompt
# source (see ``_validate_prompt_source`` in mlflow/server/handlers.py). The UI hardcodes the same
# "dummy-source" literal in mlflow/server/js/src/experiment-tracking/pages/prompts/api.ts.
_CLIENT_PROMPT_SOURCE_PLACEHOLDER = "dummy-source"
_STORE_PROMPT_SOURCE_PLACEHOLDER = "prompt-template"
_PROMPT_SOURCE_PLACEHOLDERS = frozenset({
    _CLIENT_PROMPT_SOURCE_PLACEHOLDER,
    _STORE_PROMPT_SOURCE_PLACEHOLDER,
})

# Unity Catalog tags cannot contain dots
PROMPT_TYPE_TAG_KEY = "_mlflow_prompt_type"
RESPONSE_FORMAT_TAG_KEY = "_mlflow_prompt_response_format"
PROMPT_MODEL_CONFIG_TAG_KEY = "_mlflow_prompt_model_config"

# Prompt types
PROMPT_TYPE_TEXT = "text"
PROMPT_TYPE_CHAT = "chat"

# A special tag to store associated run IDs for prompts
PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY = "mlflow.prompt.associatedRunIds"

# A special tag to store associated experiment IDs for prompts (comma-separated list)
# Using underscore prefix for Unity Catalog compatibility (UC tags cannot contain dots)
PROMPT_EXPERIMENT_IDS_TAG_KEY = "_mlflow_experiment_ids"

PROMPT_TEMPLATE_VARIABLE_PATTERN = re.compile(
    r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\}\}"
)

PROMPT_TEXT_DISPLAY_LIMIT = 30

# Alphanumeric, underscore, hyphen, and dot are allowed in prompt name
PROMPT_NAME_RULE = re.compile(r"^[a-zA-Z0-9_.-]+$")
