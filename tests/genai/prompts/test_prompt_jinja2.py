import pytest
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.tracking import MlflowClient
from mlflow.prompt.constants import PROMPT_TYPE_JINJA2

# 1. Basic Jinja2 rendering test
def test_jinja2_prompt_basic_rendering():
    p = PromptVersion("jinja-basic", 1, "Hello {% if name %}{{ name }}{% else %}Guest{% endif %}")
    assert p.format(name="Alice") == "Hello Alice"
    assert p.format() == "Hello Guest"

# 2. Loop (for-statement) rendering test
def test_jinja2_prompt_loop_rendering():
    template = "Fruits: {% for f in fruits %}{{ f }} {% endfor %}"
    p = PromptVersion("jinja-loop", 1, template)
    result = p.format(fruits=["apple", "banana", "cherry"])
    assert "apple" in result and "banana" in result and "cherry" in result

# 3. Sandbox disabled test
def test_jinja2_prompt_no_sandbox():
    p = PromptVersion("jinja-nosandbox", 1, "{{ 2 * 3 }}")
    assert p.format(use_jinja_sandbox=False) == "6"

# 4. Test that MlflowClient.register_prompt() detects Jinja2 templates
def test_client_register_prompt_detects_jinja2(monkeypatch):
    client = MlflowClient()

    created_versions = []

    def mock_create_model_version(name, description, source, tags):
        created_versions.append(tags)

        class Dummy:
            def __init__(self, name, description, tags):
                self.version = 1
                self.name = name
                self.description = description
                self.tags = tags
                self.aliases = []
                self.creation_timestamp = None
                self.last_updated_timestamp = None
                self.user_id = None

        return Dummy(name, description, tags)

    monkeypatch.setattr(client._get_registry_client(), "create_model_version", mock_create_model_version)

    template = "Hello {{ user }} from {% if country %}{{ country }}{% else %}somewhere{% endif %}"
    client.register_prompt(name="jinja-detect", template=template)

    tags = created_versions[0]
    assert tags["mlflow.prompt.type"] == PROMPT_TYPE_JINJA2
    assert "mlflow.prompt.text" in tags

# 5. Fallback for plain text templates
def test_client_register_prompt_plain_text(monkeypatch):
    client = MlflowClient()

    created_versions = []

    def mock_create_model_version(name, description, source, tags):
        created_versions.append(tags)

        class Dummy:
            def __init__(self, name, description, tags):
                self.version = 1
                self.name = name
                self.description = description
                self.tags = tags
                self.aliases = []
                self.creation_timestamp = None
                self.last_updated_timestamp = None
                self.user_id = None

        return Dummy(name, description, tags)

    monkeypatch.setattr(client._get_registry_client(), "create_model_version", mock_create_model_version)

    client.register_prompt(name="plain-detect", template="Hello {{name}}")

    tags = created_versions[0]
    assert tags["mlflow.prompt.type"] == "text"   
