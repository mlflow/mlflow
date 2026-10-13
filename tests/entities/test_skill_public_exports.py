from mlflow import entities


def test_skill_entities_exported():
    for name in [
        "Skill",
        "SkillStatus",
        "SkillVersion",
        "SkillSourceType",
        "GitSource",
        "MlflowSource",
        "OCISource",
        "ZipSource",
    ]:
        assert name in entities.__all__, f"{name} missing from mlflow.entities.__all__"
        assert getattr(entities, name) is not None


def test_import_from_entities():
    from mlflow.entities import GitSource, MlflowSource, Skill  # noqa: F401
