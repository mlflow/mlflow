# A special tag in RegisteredModel to indicate that it is a prompt
import re

IS_PROMPT_TAG_KEY = "mlflow.prompt.is_prompt"
# A special tag in ModelVersion to store the prompt text
PROMPT_TEXT_TAG_KEY = "mlflow.prompt.text"

LINKED_PROMPTS_TAG_KEY = "mlflow.linkedPrompts"

# A special tag to store associated run IDs for prompts
PROMPT_ASSOCIATED_RUN_IDS_TAG_KEY = "mlflow.prompt.associatedRunIds"

PROMPT_TEMPLATE_VARIABLE_PATTERN = re.compile(
    r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\}\}"
)

PROMPT_TEXT_DISPLAY_LIMIT = 30

# Alphanumeric, underscore, hyphen, and dot are allowed in prompt name
PROMPT_NAME_RULE = re.compile(r"^[a-zA-Z0-9_.-]+$")
