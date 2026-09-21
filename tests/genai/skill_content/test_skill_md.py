import os

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.skill_md import (
    SKILL_MANIFEST_FILE,
    SkillManifest,
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
    assert manifest.path == root


@pytest.mark.parametrize(
    "frontmatter",
    [
        "keywords: [review, quality]",
        "keywords: ' x, y ,,z '",
        "keywords: {a: b}",
        "keywords: [review, 2]",
        "keywords: [[nested]]",
        "metadata:\n  keywords: [[x]]",
        "metadata: not-a-mapping",
        "license: MIT\nallowed-tools: [Bash]",
    ],
)
def test_inspect_skill_dir_ignores_other_frontmatter_keys(tmp_path, frontmatter):
    # Skills have no keywords; a SKILL.md that carries such a field for other tooling, in any
    # shape, is inspected like one without it.
    content = f"---\nname: demo\ndescription: Demo skill\n{frontmatter}\n---\n"
    manifest = inspect_skill_dir(_skill_dir(tmp_path, "demo", content))
    assert manifest == SkillManifest(name="demo", description="Demo skill", path=manifest.path)
    assert not hasattr(manifest, "keywords")


def test_inspect_skill_dir_requires_declared_name(tmp_path):
    # Fetched content lands in an arbitrary directory such as ``content``, so the directory
    # name is never an acceptable identity.
    root = _skill_dir(tmp_path, "content", "# no frontmatter\n")
    with pytest.raises(MlflowException, match="must declare a 'name'"):
        inspect_skill_dir(root)
    manifest = inspect_skill_dir(root, fallback_name="legacy-skill")
    assert manifest.name == "legacy-skill"
    assert manifest.description is None


def test_inspect_skill_dir_declared_name_beats_fallback(tmp_path):
    root = _skill_dir(tmp_path, "dir", "---\nname: declared\n---\n")
    assert inspect_skill_dir(root, fallback_name="other").name == "declared"


def test_inspect_skill_dir_accepts_utf8_bom(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    (root / SKILL_MANIFEST_FILE).write_bytes(b"\xef\xbb\xbf---\nname: demo\ndescription: d\n---\n")
    manifest = inspect_skill_dir(root)
    assert (manifest.name, manifest.description) == ("demo", "d")


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
    ],
)
def test_inspect_skill_dir_field_type_errors(tmp_path, content, message):
    root = _skill_dir(tmp_path, "demo", content)
    with pytest.raises(MlflowException, match=message):
        inspect_skill_dir(root)


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="permission bits are not enforced here"
)
def test_inspect_skill_dir_unreadable_manifest(tmp_path):
    root = _skill_dir(tmp_path, "demo", "---\nname: demo\n---\n")
    (root / SKILL_MANIFEST_FILE).chmod(0)
    with pytest.raises(MlflowException, match="Cannot read skill content") as exc:
        inspect_skill_dir(root)
    assert exc.value.error_code == "PERMISSION_DENIED"


@pytest.mark.parametrize(
    "content",
    [
        "---\nbase: &b {a: 1}\nname: demo\nextra: *b\n---\n",
        "---\nbase: &b {a: 1}\nname: demo\nmerged:\n  <<: *b\n---\n",
        # Nested merges double the intermediate mapping at each level; SafeLoader would spend
        # minutes on a few hundred bytes of this.
        "---\nname: demo\n"
        + "".join(f"l{i}: &l{i} [*l{i - 1}, *l{i - 1}]\n" for i in range(1, 12)).replace("*l0", "x")
        + "---\n",
    ],
)
def test_parse_skill_md_rejects_yaml_aliases(content):
    with pytest.raises(MlflowException, match="aliases and merge keys are not allowed"):
        parse_skill_md(content)


@pytest.mark.parametrize(
    "frontmatter",
    [
        # A merge key can pull fields from an inline mapping without any alias event.
        "<<: {description: merged}",
        # An explicit merge tag is applied by SafeLoader whatever the key's value or style.
        '!!merge "x": {description: merged}',
        "!!merge x: {description: merged}",
        "!<tag:yaml.org,2002:merge> x: {description: merged}",
    ],
)
def test_parse_skill_md_rejects_inline_merge_keys(frontmatter):
    with pytest.raises(MlflowException, match="aliases and merge keys are not allowed"):
        parse_skill_md(f"---\nname: demo\n{frontmatter}\n---\n")
    # A quoted "<<" is an ordinary key, not a merge.
    metadata, _ = parse_skill_md('---\nname: demo\n"<<": literal\n---\n')
    assert metadata == {"name": "demo", "<<": "literal"}
