from mlflow import entities


def test_skill_entities_exported():
    for name in [
        "Skill",
        "SkillStatus",
        "SkillVersion",
        "SkillSourceType",
        "GitSource",
        "OCISource",
        "ZipSource",
        "AgentPlugin",
        "AgentPluginVersion",
    ]:
        assert name in entities.__all__, f"{name} missing from mlflow.entities.__all__"
        assert getattr(entities, name) is not None


def test_import_from_entities():
    from mlflow.entities import AgentPlugin, GitSource, Skill  # noqa: F401
