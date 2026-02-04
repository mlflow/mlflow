# A special tag in RegisteredModel to indicate that it is a prompt
import re

IS_PROMPT_TAG_KEY = "mlflow.prompt.is_prompt"
# A special tag in ModelVersion to store the prompt text
PROMPT_TEXT_TAG_KEY = "mlflow.prompt.text"

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
