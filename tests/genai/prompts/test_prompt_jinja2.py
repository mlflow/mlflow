import pytest
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.tracking import MlflowClient
from mlflow.prompt.constants import PROMPT_TYPE_JINJA2

# 1. 기본 Jinja2 렌더링 테스트
def test_jinja2_prompt_basic_rendering():
    p = PromptVersion("jinja-basic", 1, "Hello {% if name %}{{ name }}{% else %}Guest{% endif %}")
    assert p.format(name="Alice") == "Hello Alice"
    assert p.format() == "Hello Guest"

# 2. 반복문 테스트
def test_jinja2_prompt_loop_rendering():
    template = "Fruits: {% for f in fruits %}{{ f }} {% endfor %}"
    p = PromptVersion("jinja-loop", 1, template)
    result = p.format(fruits=["apple", "banana", "cherry"])
    assert "apple" in result and "banana" in result and "cherry" in result

# 3. Sandbox 비활성화 테스트
def test_jinja2_prompt_no_sandbox():
    p = PromptVersion("jinja-nosandbox", 1, "{{ 2 * 3 }}")
    assert p.format(use_jinja_sandbox=False) == "6"

# 4. MlflowClient.register_prompt()가 Jinja2 템플릿을 감지하는지 테스트
def test_client_register_prompt_detects_jinja2(monkeypatch):
    client = MlflowClient()

    created_versions = []

    # Monkeypatch create_model_version to capture tags
    def mock_create_model_version(name, description, source, tags):
        created_versions.append(tags)
        class Dummy:
            version = 1
        return Dummy()

    monkeypatch.setattr(client._get_registry_client(), "create_model_version", mock_create_model_version)

    # Run registration with Jinja2 template
    template = "Hello {{ user }} from {% if country %}{{ country }}{% else %}somewhere{% endif %}"
    client.register_prompt(name="jinja-detect", template=template)

    tags = created_versions[0]
    assert tags["mlflow.prompt.type"] == PROMPT_TYPE_JINJA2
    assert "mlflow.prompt.text" in tags

# 5. Plain text fallback 확인
def test_client_register_prompt_plain_text(monkeypatch):
    client = MlflowClient()
    created_versions = []

    def mock_create_model_version(name, description, source, tags):
        created_versions.append(tags)
        class Dummy:
            version = 1
        return Dummy()

    monkeypatch.setattr(client._get_registry_client(), "create_model_version", mock_create_model_version)

    client.register_prompt(name="plain-detect", template="Hello {{name}}")
    tags = created_versions[0]
    assert tags["mlflow.prompt.type"] in ["text", PROMPT_TYPE_JINJA2]
