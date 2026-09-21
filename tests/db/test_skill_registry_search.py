# Cross-dialect (Docker matrix) tests for the RFC-0008 skill registry search helpers.
# They run against every backend in the MLflow DB test matrix (SQLite, PostgreSQL,
# MySQL, MSSQL) and cover the behavior that depends on the engine: string comparison
# case sensitivity, correlated EXISTS and GROUP BY/HAVING subqueries, SemVer ordering,
# and ordering of rows that tie on the requested key.
#
# Each test mirrors a SQLite-only test in tests/store/tracking/test_skill_registry_filter.py
# and names its counterpart. Sessions come from the store's own ManagedSessionMaker, so
# SQLite runs with the same pragmas as production (including case_sensitive_like).
#
# NOTE: all tests here share one database, so each test works in its own organization
# (the `org` fixture) and filters every query by it.

import uuid
from pathlib import Path

import pytest

from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.store.tracking.dbmodels.models import (
    SqlAgentPlugin,
    SqlAgentPluginVersion,
    SqlAgentPluginVersionMember,
    SqlSkill,
    SqlSkillTag,
    SqlSkillVersion,
    SqlSkillVersionTag,
)
from mlflow.store.tracking.skill_registry_filter import (
    apply_member_name_filter,
    apply_skill_registry_filters,
    paginate_results,
    parse_skill_registry_order_by,
)
from mlflow.store.tracking.skill_registry_pagination import SkillRegistryPaginationToken
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.search_utils import SearchSkillUtils, SearchSkillVersionUtils

pytestmark = pytest.mark.notrackingurimock


@pytest.fixture
def store(tmp_path: Path):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    store = SqlAlchemyStore(MLFLOW_TRACKING_URI.get(), artifact_uri.as_uri())
    try:
        yield store
    finally:
        store._dispose_engine()


@pytest.fixture
def org():
    return f"search-{uuid.uuid4().hex[:12]}"


def _add_skill(session, org, name, *, description=None, search_text=None, status=None, tags=None):
    session.add(
        SqlSkill(
            workspace="default",
            organization=org,
            name=name,
            description=description,
            search_text=search_text,
        )
    )
    if status is not None:
        session.add(
            SqlSkillVersion(
                workspace="default", organization=org, name=name, version=1, status=status
            )
        )
    for key, value in (tags or {}).items():
        session.add(
            SqlSkillTag(workspace="default", organization=org, name=name, key=key, value=value)
        )


def _add_plugin_version(session, org, plugin, version, members, status="active"):
    session.add(
        SqlAgentPluginVersion(
            workspace="default",
            organization=org,
            name=plugin,
            version=version,
            status=status,
            plugin_json={"name": plugin, "version": version},
        )
    )
    for member in members:
        session.add(
            SqlAgentPluginVersionMember(
                plugin_workspace="default",
                plugin_organization=org,
                plugin_name=plugin,
                plugin_version=version,
                member_organization=org,
                member_name=member,
                member_version=1,
            )
        )


def _skill_query(session, org, store, filter_string):
    return apply_skill_registry_filters(
        session.query(SqlSkill).filter(SqlSkill.organization == org),
        SearchSkillUtils.parse_search_filter(filter_string),
        {
            "name": SqlSkill.name,
            "description": SqlSkill.description,
            "search_text": SqlSkill.search_text,
            "status": SqlSkill.resolved_status_expression(),
        },
        SqlSkill,
        SqlSkillTag,
        tag_join_keys=["workspace", "organization", "name"],
        dialect=store.engine.dialect.name,
    )


def _skill_names(store, org, filter_string):
    with store.ManagedSessionMaker() as session:
        return sorted(skill.name for skill in _skill_query(session, org, store, filter_string))


# Counterpart: test_filter_computed_status_stays_case_sensitive (SQL text only on SQLite).
@pytest.mark.parametrize(
    ("filter_string", "matches"),
    [
        # Mapped column.
        ("search_text LIKE '%Reviews%'", True),
        ("search_text LIKE '%reviews%'", False),
        ("search_text ILIKE '%REVIEWS%'", True),
        # Computed expression (resolved parent status).
        ("status = 'active'", True),
        ("status = 'Active'", False),
        ("status LIKE 'act%'", True),
        ("status LIKE 'Act%'", False),
        ("status ILIKE 'ACT%'", True),
    ],
)
def test_db_backend_like_is_case_sensitive_and_ilike_is_not(store, org, filter_string, matches):
    with store.ManagedSessionMaker(read_only=False) as session:
        _add_skill(
            session, org, "code-review", search_text="code-review Reviews PRs", status="active"
        )

    assert _skill_names(store, org, filter_string) == (["code-review"] if matches else [])


@pytest.mark.parametrize(
    ("filter_string", "matches"),
    [
        ("status ILIKE 'ACT%'", True),
        ("status LIKE 'Act%'", False),
        ("status = 'Active'", False),
    ],
)
def test_db_backend_ilike_on_binary_collated_expression(store, org, filter_string, matches):
    # A binary collation makes plain LIKE case-sensitive, so ILIKE must lower
    # both sides rather than rely on the default collation.
    if store.engine.dialect.name != "mysql":
        pytest.skip("Only MySQL chooses case sensitivity by collation for computed LIKE")
    with store.ManagedSessionMaker(read_only=False) as session:
        _add_skill(session, org, "code-review", status="active")

    status = SqlSkill.resolved_status_expression().collate("utf8mb4_bin")
    with store.ManagedSessionMaker() as session:
        query = apply_skill_registry_filters(
            session.query(SqlSkill).filter(SqlSkill.organization == org),
            SearchSkillUtils.parse_search_filter(filter_string),
            {"status": status},
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        assert [skill.name for skill in query] == (["code-review"] if matches else [])


# Counterpart: test_filter_by_computed_status and test_filter_by_status_list.
def test_db_backend_status_list_on_computed_status(store, org):
    with store.ManagedSessionMaker(read_only=False) as session:
        _add_skill(session, org, "active-skill", status="active")
        _add_skill(session, org, "draft-skill", status="draft")
        _add_skill(session, org, "deprecated-skill", status="deprecated")

    assert _skill_names(store, org, "status IN ('active', 'draft')") == [
        "active-skill",
        "draft-skill",
    ]
    assert _skill_names(store, org, "status NOT IN ('active')") == [
        "deprecated-skill",
        "draft-skill",
    ]


# Counterpart: test_filter_by_multiple_tags and test_filter_by_tag_operators.
def test_db_backend_tag_filters_intersect_per_key(store, org):
    with store.ManagedSessionMaker(read_only=False) as session:
        _add_skill(session, org, "a", tags={"team": "platform", "priority": "high"})
        _add_skill(session, org, "b", tags={"team": "platform"})
        _add_skill(session, org, "c", tags={"team": "platform-ml"})
        _add_skill(session, org, "d", tags={"team": "infra"})

    assert _skill_names(store, org, "tags.team = 'platform' AND tags.priority = 'high'") == ["a"]
    assert _skill_names(store, org, "tags.team LIKE 'plat%' AND tags.team != 'platform'") == ["c"]


# Counterparts: test_filter_by_timestamp_columns and test_filter_by_skill_version_number.
# Numeric filter values are bound as integers: PostgreSQL rejects comparing a
# BIGINT column to a string parameter, and the other engines only coerce it by
# accident.
@pytest.mark.parametrize(
    ("filter_string", "expected"),
    [
        ("created_at > 2000", ["newer"]),
        ("created_at < 2000", ["older"]),
        ("created_at = 1000", ["older"]),
        ("last_updated_at >= 3000", ["newer"]),
    ],
)
def test_db_backend_timestamp_filters(store, org, filter_string, expected):
    with store.ManagedSessionMaker(read_only=False) as session:
        for name, stamp in [("older", 1000), ("newer", 3000)]:
            _add_skill(session, org, name)
            skill = (
                session
                .query(SqlSkill)
                .filter(SqlSkill.organization == org, SqlSkill.name == name)
                .one()
            )
            skill.created_at = stamp
            skill.last_updated_at = stamp

    with store.ManagedSessionMaker() as session:
        query = apply_skill_registry_filters(
            session.query(SqlSkill).filter(SqlSkill.organization == org),
            SearchSkillUtils.parse_search_filter(filter_string),
            {"created_at": SqlSkill.created_at, "last_updated_at": SqlSkill.last_updated_at},
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        assert sorted(skill.name for skill in query) == expected


# Version 10 sorts before 3 as a string, so this also shows the comparison is
# numeric rather than lexicographic on every engine.
@pytest.mark.parametrize(
    ("filter_string", "expected"),
    [
        ("version >= 3", [3, 10]),
        ("version > 3", [10]),
        ("version < 3", [1]),
        ("version != 10", [1, 3]),
    ],
)
def test_db_backend_skill_version_number_filters(store, org, filter_string, expected):
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(SqlSkill(workspace="default", organization=org, name="code-review"))
        for version in (1, 3, 10):
            session.add(
                SqlSkillVersion(
                    workspace="default",
                    organization=org,
                    name="code-review",
                    version=version,
                    status="active",
                )
            )

    with store.ManagedSessionMaker() as session:
        query = apply_skill_registry_filters(
            session.query(SqlSkillVersion).filter(SqlSkillVersion.organization == org),
            SearchSkillVersionUtils.parse_search_filter(filter_string),
            {"version": SqlSkillVersion.version},
            SqlSkillVersion,
            SqlSkillVersionTag,
            tag_join_keys=["workspace", "organization", "name", "version"],
            dialect=store.engine.dialect.name,
        )
        assert sorted(row.version for row in query) == expected


# Counterparts: test_member_name_excludes_version_with_deleted_sibling,
# test_member_name_keeps_version_with_deprecated_sibling,
# test_member_name_matches_older_eligible_version_when_newer_is_withdrawn, and
# test_member_name_excludes_deleted_plugin_version.
def test_db_backend_member_search_excludes_withdrawn_versions(store, org):
    with store.ManagedSessionMaker(read_only=False) as session:
        _add_skill(session, org, "code-review", status="active")
        _add_skill(session, org, "unsafe-helper", status="deleted")
        _add_skill(session, org, "old-helper", status="deprecated")
        for plugin in ["withdrawn", "deprecated-sibling", "mixed", "deleted-version"]:
            session.add(SqlAgentPlugin(workspace="default", organization=org, name=plugin))
        _add_plugin_version(session, org, "withdrawn", "1.0.0", ["code-review", "unsafe-helper"])
        _add_plugin_version(
            session, org, "deprecated-sibling", "1.0.0", ["code-review", "old-helper"]
        )
        _add_plugin_version(session, org, "mixed", "1.0.0", ["code-review"])
        _add_plugin_version(session, org, "mixed", "2.0.0", ["code-review", "unsafe-helper"])
        _add_plugin_version(
            session, org, "deleted-version", "1.0.0", ["code-review"], status="deleted"
        )

    with store.ManagedSessionMaker() as session:
        query = apply_member_name_filter(
            session.query(SqlAgentPlugin).filter(SqlAgentPlugin.organization == org),
            "code-review",
            dialect=store.engine.dialect.name,
        )
        assert sorted(plugin.name for plugin in query) == ["deprecated-sibling", "mixed"]


# Counterpart: test_order_by_multi_column_key_sorts_semver_by_precedence.
@pytest.mark.parametrize("direction", ["DESC", "ASC"])
def test_db_backend_semver_order_by_precedence(store, org, direction):
    by_precedence = ["1.0.0-alpha", "1.0.0-beta", "1.0.0", "2.0.0", "9.1.0", "10.0.0"]
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(SqlAgentPlugin(workspace="default", organization=org, name="p"))
        for version in ["2.0.0", "1.0.0", "10.0.0", "1.0.0-beta", "9.1.0", "1.0.0-alpha"]:
            _add_plugin_version(session, org, "p", version, [])

    clauses = parse_skill_registry_order_by(
        [f"version {direction}"],
        valid_keys={"version"},
        column_map={
            "version": (
                SqlAgentPluginVersion.version_major,
                SqlAgentPluginVersion.version_minor,
                SqlAgentPluginVersion.version_patch,
                SqlAgentPluginVersion.version_prerelease_sort_key,
            )
        },
        default_tiebreakers=[],
    )
    with store.ManagedSessionMaker() as session:
        ordered = [
            version.version
            for version in session
            .query(SqlAgentPluginVersion)
            .filter(SqlAgentPluginVersion.organization == org)
            .order_by(*clauses)
        ]
    assert ordered == (by_precedence[::-1] if direction == "DESC" else by_precedence)


# Counterpart: test_pagination_through_helpers_returns_every_row_once. Engines such as
# PostgreSQL return tied rows in no guaranteed order, so this is where the tiebreaker
# is what keeps rows from being repeated or skipped across pages.
def test_db_backend_pagination_with_tied_sort_key(store, org):
    names = [f"skill-{i}" for i in range(7)]
    with store.ManagedSessionMaker(read_only=False) as session:
        for name in names:
            _add_skill(session, org, name, description="shared")

    order_by = ["description ASC"]
    collected: list[str] = []
    offset = 0
    with store.ManagedSessionMaker() as session:
        while True:
            clauses = parse_skill_registry_order_by(
                order_by,
                valid_keys={"description"},
                column_map={"description": SqlSkill.description},
                default_tiebreakers=[SqlSkill.name.desc()],
            )
            rows = (
                _skill_query(session, org, store, None)
                .order_by(*clauses)
                .offset(offset)
                .limit(4)
                .all()
            )
            page = paginate_results(
                [row.name for row in rows],
                max_results=3,
                offset=offset,
                filter_string=None,
                order_by=order_by,
                query_scope="skills",
            )
            collected.extend(page)
            if page.token is None:
                break
            offset = SkillRegistryPaginationToken.decode(page.token).offset

    assert collected == sorted(names, reverse=True)
