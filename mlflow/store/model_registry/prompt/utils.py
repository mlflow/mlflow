from typing import Optional

from mlflow.entities.model_registry.prompt import IS_PROMPT_TAG_KEY


def add_prompt_filter_string(
    filter_string: Optional[str], is_prompt: bool = False
) -> Optional[str]:
    """
    Additional filter string to include/exclude prompts from the result.
    By default, exclude prompts from the result.
    """
    if IS_PROMPT_TAG_KEY not in (filter_string or ""):
        prompt_filter_query = (
            f"tag.`{IS_PROMPT_TAG_KEY}` = 'true'"
            if is_prompt
            else f"tag.`{IS_PROMPT_TAG_KEY}` != 'true'"
        )
        if filter_string:
            filter_string = f"{filter_string} AND {prompt_filter_query}"
        else:
            filter_string = prompt_filter_query
    return filter_string
