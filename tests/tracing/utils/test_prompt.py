import json

import pytest

from mlflow.entities.model_registry import PromptVersion
from mlflow.exceptions import MlflowException
from mlflow.tracing.utils.prompt import update_linked_prompts_tag


def test_update_linked_prompts_tag():
    pv1 = PromptVersion(name="test_prompt", version=1, template="Test template")
    updated_tag_value = update_linked_prompts_tag(None, [pv1])
    assert json.loads(updated_tag_value) == [{"name": "test_prompt", "version": "1"}]

    # Adding multiple prompts to the same trace
    pv2 = PromptVersion(name="test_prompt", version=2, template="Test template 2")
    pv3 = PromptVersion(name="test_prompt_3", version=1, template="Test template 3")
    updated_tag_value = update_linked_prompts_tag(updated_tag_value, [pv2, pv3])
    assert json.loads(updated_tag_value) == [
        {"name": "test_prompt", "version": "1"},
        {"name": "test_prompt", "version": "2"},
        {"name": "test_prompt_3", "version": "1"},
    ]

    # Registering the same prompt should not add it again
    updated_tag_value = update_linked_prompts_tag(updated_tag_value, [pv1])
    assert json.loads(updated_tag_value) == [
        {"name": "test_prompt", "version": "1"},
        {"name": "test_prompt", "version": "2"},
        {"name": "test_prompt_3", "version": "1"},
    ]


def test_update_linked_prompts_tag_invalid_current_tag():
    prompt_version = PromptVersion(name="test_prompt", version=1, template="Test template")

    with pytest.raises(MlflowException, match="Invalid JSON format for 'mlflow.linkedPrompts' tag"):
        update_linked_prompts_tag("invalid json", [prompt_version])

    with pytest.raises(MlflowException, match="Invalid format for 'mlflow.linkedPrompts' tag"):
        update_linked_prompts_tag(json.dumps({"not": "a list"}), [prompt_version])
