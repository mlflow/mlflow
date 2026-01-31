import pytest

from mlflow.demo.base import DEMO_PROMPT_PREFIX, DemoFeature, DemoResult
from mlflow.demo.data import DEMO_PROMPTS
from mlflow.demo.generators.prompts import PromptsDemoGenerator
from mlflow.genai.prompts import load_prompt, search_prompts


@pytest.fixture
def prompts_generator():
    generator = PromptsDemoGenerator()
    original_version = generator.version
    yield generator
    PromptsDemoGenerator.version = original_version


def test_generator_attributes():
    generator = PromptsDemoGenerator()
    assert generator.name == DemoFeature.PROMPTS
    assert generator.version == 2


def test_data_exists_false_when_no_prompts():
    generator = PromptsDemoGenerator()
    assert generator._data_exists() is False


def test_generate_creates_prompts():
    generator = PromptsDemoGenerator()
    result = generator.generate()

    assert isinstance(result, DemoResult)
    assert result.feature == DemoFeature.PROMPTS
    assert any("prompts:" in e for e in result.entity_ids)
    assert any("versions:" in e for e in result.entity_ids)


def test_generate_creates_expected_prompts():
    generator = PromptsDemoGenerator()
    generator.generate()

    prompts = search_prompts(
        filter_string=f"name LIKE '{DEMO_PROMPT_PREFIX}.%'",
        max_results=100,
    )

    assert len(prompts) == len(DEMO_PROMPTS)

    prompt_names = {p.name for p in prompts}
    expected_names = {prompt_def.name for prompt_def in DEMO_PROMPTS}
    assert prompt_names == expected_names


def test_prompts_have_multiple_versions():
    generator = PromptsDemoGenerator()
    generator.generate()

    for prompt_def in DEMO_PROMPTS:
        expected_versions = len(prompt_def.versions)
        prompt = load_prompt(prompt_def.name, version=expected_versions)
        assert prompt is not None
        assert prompt.version == expected_versions


def test_prompts_have_version_aliases():
    generator = PromptsDemoGenerator()
    generator.generate()

    for prompt_def in DEMO_PROMPTS:
        for version_num, version_def in enumerate(prompt_def.versions, start=1):
            if version_def.aliases:
                prompt = load_prompt(f"prompts:/{prompt_def.name}@{version_def.aliases[0]}")
                assert prompt.version == version_num


def test_data_exists_true_after_generate():
    generator = PromptsDemoGenerator()
    assert generator._data_exists() is False

    generator.generate()

    assert generator._data_exists() is True


def test_delete_demo_removes_prompts():
    generator = PromptsDemoGenerator()
    generator.generate()
    assert generator._data_exists() is True

    generator.delete_demo()

    assert generator._data_exists() is False


def test_prompts_have_demo_tag():
    generator = PromptsDemoGenerator()
    generator.generate()

    for prompt_def in DEMO_PROMPTS:
        prompt = load_prompt(prompt_def.name, version=1)
        assert prompt.tags.get("demo") == "true"


def test_is_generated_checks_version(prompts_generator):
    prompts_generator.generate()
    prompts_generator.store_version()

    assert prompts_generator.is_generated() is True

    PromptsDemoGenerator.version = 99
    fresh_generator = PromptsDemoGenerator()
    assert fresh_generator.is_generated() is False


def test_prompt_templates_are_valid():
    generator = PromptsDemoGenerator()
    generator.generate()

    for prompt_def in DEMO_PROMPTS:
        latest_version = len(prompt_def.versions)
        prompt = load_prompt(prompt_def.name, version=latest_version)

        assert prompt.template is not None
        if isinstance(prompt.template, str):
            assert len(prompt.template) > 0
        else:
            assert len(prompt.template) > 0
            for msg in prompt.template:
                assert "role" in msg
                assert "content" in msg


def test_demo_prompt_definitions():
    assert len(DEMO_PROMPTS) == 3
    for prompt_def in DEMO_PROMPTS:
        assert prompt_def.name.startswith(DEMO_PROMPT_PREFIX)
        assert len(prompt_def.versions) >= 3
        for version_def in prompt_def.versions:
            assert version_def.template is not None
            assert version_def.commit_message is not None
