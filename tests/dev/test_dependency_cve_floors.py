"""
Regression test for mlflow#23061.

Asserts that the minimum-version floors for `mlflow-skinny` core
dependencies in `requirements/skinny-requirements.yaml` and the optional
`mcp` extra in `dev/pyproject.py` exclude versions affected by the CVEs
listed in the issue. Without this test, a future "loosen the floor" change
would silently re-admit the vulnerable versions.

Each entry maps a dependency to the highest *vulnerable* version. The test
fails if the declared floor permits installing that version. This intentionally
encodes the security floor as data, so when a CVE is resolved upstream the
maintainer can simply delete the entry.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]


def _yaml_dep_specifier(yaml_path: Path, name: str) -> SpecifierSet:
    """Reconstruct the pip specifier (e.g. ">=3.1.47,<4") from one of the
    YAML requirement files used as the source of truth for `dev/pyproject.py`.
    """
    spec_data = yaml.safe_load(yaml_path.read_text())
    entry = spec_data[name]
    parts = []
    if "minimum" in entry:
        parts.append(f">={entry['minimum']}")
    if "max_major_version" in entry:
        parts.append(f"<{int(entry['max_major_version']) + 1}")
    return SpecifierSet(",".join(parts))


# Dependencies pinned in `requirements/skinny-requirements.yaml`.
# (dep_name, last_vulnerable_version_that_must_be_excluded, advisory_ref)
SKINNY_DEPS_AGAINST_CVES: list[tuple[str, str, str]] = [
    # GHSA-rpm5-65cw-6hj4, GHSA-x2qx-6953-8485 — fixed in 3.1.47.
    ("gitpython", "3.1.46", "GHSA-rpm5-65cw-6hj4"),
    # GHSA-mf9w-mj56-hr94 — fixed in 1.2.2.
    ("python-dotenv", "1.2.1", "GHSA-mf9w-mj56-hr94"),
    # GHSA-gc5v-m9x4-r6x2 — fixed in 2.33.0.
    ("requests", "2.32.5", "GHSA-gc5v-m9x4-r6x2"),
]


def test_skinny_requirements_floors_exclude_known_cves():
    yaml_path = REPO_ROOT / "requirements" / "skinny-requirements.yaml"
    for name, vuln_version, advisory in SKINNY_DEPS_AGAINST_CVES:
        spec = _yaml_dep_specifier(yaml_path, name)
        assert Version(vuln_version) not in spec, (
            f"{name} floor in {yaml_path.name} permits installing {vuln_version}, "
            f"which is affected by {advisory}. Current specifier: {spec}."
        )


def test_skinny_requirements_pyproject_match_yaml():
    """Belt-and-braces: the generated `pyproject.toml` must agree with
    the YAML source of truth. If `dev/pyproject.py` was not re-run after
    editing the YAML, this catches the drift.
    """
    yaml_path = REPO_ROOT / "requirements" / "skinny-requirements.yaml"
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    skinny_pyproject = (REPO_ROOT / "libs" / "skinny" / "pyproject.toml").read_text()

    for name, _, _ in SKINNY_DEPS_AGAINST_CVES:
        spec = _yaml_dep_specifier(yaml_path, name)
        # The constraint string in `pyproject.toml` is sorted by uv/taplo as
        # ">=X,<Y" but the YAML reconstruction gives the same canonical form.
        expected_min = str(spec).split(">=", 1)[1].split(",", 1)[0]
        # Each generated pyproject must reference the bumped floor verbatim.
        assert f">={expected_min}" in pyproject, (
            f"pyproject.toml does not contain `>={expected_min}` for {name}. "
            f"Did you forget to re-run `python dev/pyproject.py`?"
        )
        assert f">={expected_min}" in skinny_pyproject, (
            f"libs/skinny/pyproject.toml does not contain `>={expected_min}` for {name}."
        )


def test_mcp_extra_fastmcp_floor_excludes_known_cves():
    """`fastmcp` is pinned inline in `dev/pyproject.py` rather than the YAML.
    Verify the same floor logic for the affected versions
    (GHSA-rww4-4w9c-7733, GHSA-m8x7-r2rg-vh5g, GHSA-vv7q-7jx5-f767 — fixed in 3.2.0).
    """
    pyproject_py = (REPO_ROOT / "dev" / "pyproject.py").read_text()
    # Pull the literal "fastmcp<...,>=..." spec out of the file.
    import re

    match = re.search(r'"fastmcp([<>=,.\d ]+)"', pyproject_py)
    assert match, "could not locate fastmcp specifier in dev/pyproject.py"
    spec = SpecifierSet(match.group(1))
    last_vuln = Version("3.1.0")  # any pre-3.2.0 release should be excluded
    assert last_vuln not in spec, (
        f"fastmcp floor in dev/pyproject.py permits installing {last_vuln}, "
        f"which is affected by GHSA-rww4-4w9c-7733 / GHSA-m8x7-r2rg-vh5g / "
        f"GHSA-vv7q-7jx5-f767. Current specifier: {spec}."
    )
