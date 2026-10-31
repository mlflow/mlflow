from pathlib import Path

import pytest

from dev.check_skills import check


def create_skill(tmp_path: Path, frontmatter: str) -> Path:
    skill_dir = tmp_path / "example-skill"
    skill_dir.mkdir()
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        f"---\nname: example-skill\ndescription: Example skill\n{frontmatter}---\n",
        encoding="utf-8",
    )
    return skill_path


def test_requires_codex_policy_when_model_invocation_is_disabled(tmp_path: Path) -> None:
    skill_path = create_skill(tmp_path, "disable-model-invocation: true\n")

    assert check(skill_path) == ["missing `agents/openai.yaml` for disabled model invocation"]


@pytest.mark.parametrize(
    ("policy_contents", "expected_error"),
    [
        ("not: [valid", "malformed `agents/openai.yaml`"),
        ("{}", "must contain `policy.allow_implicit_invocation: false`"),
        ("policy: {}", "must set `policy.allow_implicit_invocation` to boolean false"),
        ("policy:\n  allow_implicit_invocation: true\n", "must set"),
        ('policy:\n  allow_implicit_invocation: "false"\n', "must set"),
    ],
)
def test_rejects_invalid_codex_policy(
    tmp_path: Path, policy_contents: str, expected_error: str
) -> None:
    skill_path = create_skill(tmp_path, "disable-model-invocation: true\n")
    policy_path = skill_path.parent / "agents" / "openai.yaml"
    policy_path.parent.mkdir()
    policy_path.write_text(policy_contents, encoding="utf-8")

    assert any(expected_error in error for error in check(skill_path))


def test_accepts_policy_only_codex_file(tmp_path: Path) -> None:
    skill_path = create_skill(tmp_path, "disable-model-invocation: true\n")
    policy_path = skill_path.parent / "agents" / "openai.yaml"
    policy_path.parent.mkdir()
    policy_path.write_text("policy:\n  allow_implicit_invocation: false\n", encoding="utf-8")

    assert check(skill_path) == []


@pytest.mark.parametrize("frontmatter", ["", "disable-model-invocation: false\n"])
def test_does_not_require_codex_policy_when_invocation_is_enabled(
    tmp_path: Path, frontmatter: str
) -> None:
    skill_path = create_skill(tmp_path, frontmatter)

    assert check(skill_path) == []
