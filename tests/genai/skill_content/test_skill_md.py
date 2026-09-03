import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.skill_md import (
    SKILL_MANIFEST_FILE,
    inspect_skill_dir,
    parse_skill_md,
)


def _skill_dir(tmp_path, name, content):
    root = tmp_path / name
    root.mkdir()
    (root / SKILL_MANIFEST_FILE).write_text(content, encoding="utf-8")
    return root


def test_parse_skill_md_with_frontmatter():
    metadata, body = parse_skill_md("---\nname: demo\ndescription: Does things\n---\n# Body\n")
    assert metadata == {"name": "demo", "description": "Does things"}
    assert body == "# Body\n"


def test_parse_skill_md_without_frontmatter():
    assert parse_skill_md("# Just a body\n") == ({}, "# Just a body\n")


def test_parse_skill_md_frontmatter_only():
    metadata, body = parse_skill_md("---\nname: demo\n---")
    assert metadata == {"name": "demo"}
    assert body == ""


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("---\nname: demo\n", "not closed"),
        ("---\nname: [\n---\nbody", "not valid YAML"),
        ("---\n- just\n- a list\n---\nbody", "must be a YAML mapping"),
    ],
)
def test_parse_skill_md_malformed(content, message):
    with pytest.raises(MlflowException, match=message):
        parse_skill_md(content)


def test_inspect_skill_dir_reads_fields(tmp_path):
    root = _skill_dir(
        tmp_path,
        "code-review",
        "---\nname: code-review\ndescription: Reviews code\nkeywords: [review, quality]\n---\n",
    )
    manifest = inspect_skill_dir(root)
    assert manifest.name == "code-review"
    assert manifest.description == "Reviews code"
    assert manifest.keywords == ("review", "quality")
    assert manifest.path == root


def test_inspect_skill_dir_keywords_from_string_and_nested_metadata(tmp_path):
    root = _skill_dir(tmp_path, "a", "---\nname: a\nkeywords: ' x, y ,,z '\n---\n")
    assert inspect_skill_dir(root).keywords == ("x", "y", "z")
    nested = _skill_dir(tmp_path, "b", "---\nname: b\nmetadata:\n  keywords: [q]\n---\n")
    assert inspect_skill_dir(nested).keywords == ("q",)


def test_inspect_skill_dir_name_falls_back_to_directory(tmp_path):
    root = _skill_dir(tmp_path, "fallback-name", "# no frontmatter\n")
    manifest = inspect_skill_dir(root)
    assert manifest.name == "fallback-name"
    assert manifest.description is None
    assert manifest.keywords == ()


@pytest.mark.parametrize("name", ["Bad_Name", "-lead", "trail-", "a--b", "x" * 65])
def test_inspect_skill_dir_invalid_name(tmp_path, name):
    root = _skill_dir(tmp_path, "dir", f"---\nname: {name}\n---\n")
    with pytest.raises(MlflowException, match="(?i)skill name"):
        inspect_skill_dir(root)


def test_inspect_skill_dir_missing_manifest(tmp_path):
    with pytest.raises(MlflowException, match="does not contain a SKILL.md"):
        inspect_skill_dir(tmp_path)


def test_inspect_skill_dir_rejects_symlinked_manifest(tmp_path):
    real = tmp_path / "real.md"
    real.write_text("---\nname: demo\n---\n")
    root = tmp_path / "demo"
    root.mkdir()
    (root / SKILL_MANIFEST_FILE).symlink_to(real)
    with pytest.raises(MlflowException, match="does not contain a SKILL.md"):
        inspect_skill_dir(root)


def test_inspect_skill_dir_rejects_invalid_utf8(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    (root / SKILL_MANIFEST_FILE).write_bytes(b"---\nname: demo\n---\n\xff\xfe")
    with pytest.raises(MlflowException, match="not valid UTF-8"):
        inspect_skill_dir(root)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("---\nname: 42\n---\n", "name must be a string"),
        ("---\nname: demo\ndescription: [a]\n---\n", "description must be a string"),
        ("---\nname: demo\nkeywords: {a: b}\n---\n", "keywords must be a list"),
        ("---\nname: demo\nkeywords: [[nested]]\n---\n", "keywords must be strings"),
    ],
)
def test_inspect_skill_dir_field_type_errors(tmp_path, content, message):
    root = _skill_dir(tmp_path, "demo", content)
    with pytest.raises(MlflowException, match=message):
        inspect_skill_dir(root)
