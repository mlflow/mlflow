from mlflow.genai import skills


def test_public_surface_exported():
    for name in [
        "Skill",
        "SkillVersion",
        "SkillStatus",
        "SkillSourceType",
        "GitSource",
        "OCISource",
        "ZipSource",
        "AgentPlugin",
        "AgentPluginVersion",
        "ParsedSkillUri",
        "ParsedAgentPluginUri",
        "parse_skill_uri",
        "format_skill_uri",
        "parse_agent_plugin_uri",
        "format_agent_plugin_uri",
    ]:
        assert name in skills.__all__, f"{name} missing from mlflow.genai.skills.__all__"
        assert getattr(skills, name) is not None


def test_uri_roundtrip_through_public_api():
    from mlflow.genai.skills import format_skill_uri, parse_skill_uri

    assert format_skill_uri(parse_skill_uri("skills:/@acme/code-review/1")) == (
        "skills:/@acme/code-review/1"
    )
