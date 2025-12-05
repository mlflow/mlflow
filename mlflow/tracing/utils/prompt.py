import json

from mlflow.entities.model_registry import PromptVersion
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import TraceTagKey


# TODO: Remove tag based linking once we migrate to LinkPromptsToTraces endpoint
def update_linked_prompts_tag(current_tag_value: str | None, prompt_versions: list[PromptVersion]):
    """
    Utility method to update linked prompts tag value with a new prompt version.

    Args:
        current_tag_value: Current JSON string value of the linked prompts tag
        prompt_versions: List of PromptVersion objects to add

    Returns:
        Updated JSON string with new entries added (avoiding duplicates)

    Raises:
        MlflowException: If current tag value has invalid JSON or format
    """
    if current_tag_value is not None:
        try:
            parsed_prompts_tag_value = json.loads(current_tag_value)
            if not isinstance(parsed_prompts_tag_value, list):
                raise MlflowException(
                    f"Invalid format for '{TraceTagKey.LINKED_PROMPTS}' tag: {current_tag_value}"
                )
        except json.JSONDecodeError:
            raise MlflowException(
                f"Invalid JSON format for '{TraceTagKey.LINKED_PROMPTS}' tag: {current_tag_value}"
            )
    else:
        parsed_prompts_tag_value = []

    new_prompt_entries = [
        {"name": prompt_version.name, "version": str(prompt_version.version)}
        for prompt_version in prompt_versions
    ]

    prompts_to_add = [p for p in new_prompt_entries if p not in parsed_prompts_tag_value]
    if not prompts_to_add:
        return current_tag_value

    parsed_prompts_tag_value.extend(prompts_to_add)
    return json.dumps(parsed_prompts_tag_value)
