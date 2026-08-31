from mlflow import genai


def test_skill_types_exported_from_genai():
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
    ]:
        assert name in genai.__all__, f"{name} missing from mlflow.genai.__all__"
        assert getattr(genai, name) is not None


def test_import_types_from_genai():
    from mlflow.genai import AgentPlugin, GitSource, Skill  # noqa: F401
