# Integration tests for the shared skill registry filter, order-by, and
# pagination helpers.  Uses a migrated SQLite database (via ``db_uri``) with
# seed data to verify that filters, tag joins, ordering, and pagination
# produce correct results against a real SQL engine.

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import mssql, mysql
from sqlalchemy.orm import Session

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import (
    SqlAgentPlugin,
    SqlAgentPluginTag,
    SqlAgentPluginVersion,
    SqlAgentPluginVersionMember,
    SqlAgentPluginVersionTag,
    SqlSkill,
    SqlSkillTag,
    SqlSkillVersion,
    SqlSkillVersionTag,
)
from mlflow.store.tracking.skill_registry_filter import (
    _get_comparison_func,
    apply_member_name_filter,
    apply_skill_registry_filters,
    paginate_results,
    parse_skill_registry_order_by,
)
from mlflow.store.tracking.skill_registry_pagination import (
    SkillRegistryPaginationToken,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.search_utils import (
    SearchAgentPluginUtils,
    SearchAgentPluginVersionUtils,
    SearchSkillUtils,
    SearchSkillVersionUtils,
)


@pytest.fixture
def store(tmp_path: Path, db_uri: str):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@contextmanager
def session_scope(store, *, commit=True):
    with Session(store.engine) as session:
        if store.engine.dialect.name == "sqlite":
            session.execute(sa.text("PRAGMA foreign_keys = ON"))
            # Pragmas are per connection; match production's ManagedSessionMaker
            # so LIKE does not depend on which pooled connection this session gets.
            session.execute(sa.text("PRAGMA case_sensitive_like = true"))
        yield session
        if commit:
            session.commit()


def _add_skill(
    session, *, name, organization="", description=None, search_text=None, workspace="default"
):
    skill = SqlSkill(
        workspace=workspace,
        organization=organization,
        name=name,
        description=description,
        search_text=search_text,
    )
    session.add(skill)
    return skill


def _add_skill_tag(session, *, name, organization="", key, value, workspace="default"):
    tag = SqlSkillTag(
        workspace=workspace,
        organization=organization,
        name=name,
        key=key,
        value=value,
    )
    session.add(tag)
    return tag


def _seed_skills(store):
    with session_scope(store) as session:
        _add_skill(
            session,
            name="code-review",
            organization="acme",
            description="Reviews PRs",
            search_text="code-review Reviews PRs",
        )
        _add_skill(
            session,
            name="lint-check",
            organization="acme",
            description="Runs linting",
            search_text="lint-check Runs linting",
        )
        _add_skill(
            session,
            name="deploy",
            organization="beta",
            description="Deploys apps",
            search_text="deploy Deploys apps",
        )
        _add_skill_tag(
            session,
            name="code-review",
            organization="acme",
            key="team",
            value="platform",
        )
        _add_skill_tag(
            session,
            name="code-review",
            organization="acme",
            key="priority",
            value="high",
        )
        _add_skill_tag(
            session,
            name="lint-check",
            organization="acme",
            key="team",
            value="platform",
        )
        _add_skill_tag(
            session,
            name="deploy",
            organization="beta",
            key="team",
            value="infra",
        )


def _skill_column_map():
    return {
        "name": SqlSkill.name,
        "organization": SqlSkill.organization,
        "description": SqlSkill.description,
        "search_text": SqlSkill.search_text,
    }


# ---------------------------------------------------------------------------
# Attribute filters
# ---------------------------------------------------------------------------


def test_filter_by_name_equality(store):
    _seed_skills(store)
    with session_scope(store, commit=False) as session:
        query = session.query(SqlSkill)
        parsed = SearchSkillUtils.parse_search_filter("name = 'code-review'")
        query = apply_skill_registry_filters(
            query,
            parsed,
            _skill_column_map(),
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        results = query.all()
        assert len(results) == 1
        assert results[0].name == "code-review"


def test_filter_by_name_like(store):
    _seed_skills(store)
    with session_scope(store, commit=False) as session:
        query = session.query(SqlSkill)
        parsed = SearchSkillUtils.parse_search_filter("name LIKE '%review%'")
        query = apply_skill_registry_filters(
            query,
            parsed,
            _skill_column_map(),
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        results = query.all()
        assert len(results) == 1
        assert results[0].name == "code-review"


def test_filter_by_name_lowercase_like(store):
    _seed_skills(store)
    with session_scope(store, commit=False) as session:
        query = session.query(SqlSkill)
        parsed = SearchSkillUtils.parse_search_filter("name like '%review%'")
        query = apply_skill_registry_filters(
            query,
            parsed,
            _skill_column_map(),
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        results = query.all()
        assert len(results) == 1
        assert results[0].name == "code-review"


def test_filter_preserves_organization_inside_values():
    parsed = SearchSkillUtils.parse_search_filter("description LIKE '%organization in GitHub%'")
    assert parsed[0]["value"] == "%organization in GitHub%"


def test_filter_by_organization(store):
    _seed_skills(store)
    with session_scope(store, commit=False) as session:
        query = session.query(SqlSkill)
        parsed = SearchSkillUtils.parse_search_filter("organization = 'acme'")
        query = apply_skill_registry_filters(
            query,
            parsed,
            _skill_column_map(),
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        results = query.all()
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"code-review", "lint-check"}


def test_filter_by_search_text_like(store):
    _seed_skills(store)
    with session_scope(store, commit=False) as session:
        query = session.query(SqlSkill)
        parsed = SearchSkillUtils.parse_search_filter("search_text LIKE '%linting%'")
        query = apply_skill_registry_filters(
            query,
            parsed,
            _skill_column_map(),
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        results = query.all()
        assert len(results) == 1
        assert results[0].name == "lint-check"


def test_empty_filter_returns_all(store):
    _seed_skills(store)
    with session_scope(store, commit=False) as session:
        query = session.query(SqlSkill)
        query = apply_skill_registry_filters(
            query,
            [],
            _skill_column_map(),
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        results = query.all()
        assert len(results) == 3


@pytest.mark.parametrize("dialect", ["sqlite", "postgresql", "mysql", "mssql"])
def test_filter_computed_status_builds_on_every_dialect(dialect):
    with Session() as session:
        apply_skill_registry_filters(
            session.query(SqlSkill),
            SearchSkillUtils.parse_search_filter("status = 'active'"),
            {"status": SqlSkill.resolved_status_expression()},
            SqlSkill,
            SqlSkillTag,
            ["workspace", "organization", "name"],
            dialect,
        )


@pytest.mark.parametrize(
    ("dialect_name", "dialect", "case_sensitive_sql"),
    [
        ("mysql", mysql.dialect(), "binary("),
        ("mssql", mssql.dialect(), "_CS_"),
    ],
)
@pytest.mark.parametrize(
    "filter_string", ["status = 'Active'", "status != 'Active'", "status LIKE 'Act%'"]
)
def test_filter_computed_status_stays_case_sensitive(
    dialect_name, dialect, case_sensitive_sql, filter_string
):
    # Computed expressions must compare with the same case sensitivity as columns.
    with Session() as session:
        query = apply_skill_registry_filters(
            session.query(SqlSkill),
            SearchSkillUtils.parse_search_filter(filter_string),
            {"status": SqlSkill.resolved_status_expression()},
            SqlSkill,
            SqlSkillTag,
            ["workspace", "organization", "name"],
            dialect_name,
        )
        assert case_sensitive_sql in str(query.statement.compile(dialect=dialect))


@pytest.mark.parametrize(
    "column",
    [
        SqlSkillVersion.status,
        SqlSkillVersion.status.__clause_element__().collate("utf8mb4_bin"),
    ],
    ids=["orm-column-control", "computed-case-sensitive-column"],
)
def test_mysql_ilike_preserves_case_insensitive_sql(column):
    dialect = mysql.dialect()
    condition = _get_comparison_func("ILIKE", "mysql", column)(column, "ACT%")
    actual = condition.compile(dialect=dialect)
    expected = column.ilike("ACT%").compile(dialect=dialect)
    assert str(actual) == str(expected)
    assert actual.params == expected.params


def test_filter_by_computed_status(store):
    with session_scope(store) as session:
        for name, status in [("code-review", "active"), ("lint-check", "draft")]:
            _add_skill(session, name=name, organization="acme")
            session.add(
                SqlSkillVersion(
                    workspace="default",
                    organization="acme",
                    name=name,
                    version=1,
                    status=status,
                )
            )

    with session_scope(store, commit=False) as session:
        query = apply_skill_registry_filters(
            session.query(SqlSkill),
            SearchSkillUtils.parse_search_filter("status = 'active'"),
            {"status": SqlSkill.resolved_status_expression()},
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        assert [skill.name for skill in query.all()] == ["code-review"]


@pytest.mark.parametrize(
    ("filter_string", "expected_versions"),
    [
        ("status IN ('active', 'deprecated')", {1, 3}),
        ("status NOT IN ('deleted', 'draft')", {1, 3}),
        ("status IN ('draft')", {2}),
    ],
)
def test_filter_by_status_list(store, filter_string, expected_versions):
    with session_scope(store) as session:
        _add_skill(session, name="code-review", organization="acme")
        for version, status in [(1, "active"), (2, "draft"), (3, "deprecated"), (4, "deleted")]:
            session.add(
                SqlSkillVersion(
                    workspace="default",
                    organization="acme",
                    name="code-review",
                    version=version,
                    status=status,
                )
            )

    with session_scope(store, commit=False) as session:
        query = apply_skill_registry_filters(
            session.query(SqlSkillVersion),
            SearchSkillVersionUtils.parse_search_filter(filter_string),
            {"status": SqlSkillVersion.status},
            SqlSkillVersion,
            SqlSkillVersionTag,
            tag_join_keys=["workspace", "organization", "name", "version"],
            dialect=store.engine.dialect.name,
        )
        assert {sv.version for sv in query.all()} == expected_versions


# ---------------------------------------------------------------------------
# Tag filters
# ---------------------------------------------------------------------------


def test_filter_by_single_tag(store):
    _seed_skills(store)
    with session_scope(store, commit=False) as session:
        query = session.query(SqlSkill)
        parsed = SearchSkillUtils.parse_search_filter("tags.team = 'platform'")
        query = apply_skill_registry_filters(
            query,
            parsed,
            _skill_column_map(),
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        results = query.all()
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"code-review", "lint-check"}


def test_filter_by_multiple_tags(store):
    _seed_skills(store)
    with session_scope(store, commit=False) as session:
        query = session.query(SqlSkill)
        parsed = SearchSkillUtils.parse_search_filter(
            "tags.team = 'platform' AND tags.priority = 'high'"
        )
        query = apply_skill_registry_filters(
            query,
            parsed,
            _skill_column_map(),
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        results = query.all()
        assert len(results) == 1
        assert results[0].name == "code-review"


@pytest.mark.parametrize(
    ("filter_string", "expected"),
    [
        ("tags.team LIKE 'plat%'", {"code-review", "lint-check", "ml-review"}),
        ("tags.team ILIKE 'PLAT%'", {"code-review", "lint-check", "ml-review"}),
        # != only matches skills that carry the tag, so untagged-skill is excluded.
        ("tags.team != 'platform'", {"deploy", "ml-review"}),
        # Both conditions on one key must hold for the same tag row; each alone
        # matches a different, larger set.
        ("tags.team LIKE 'plat%' AND tags.team != 'platform'", {"ml-review"}),
    ],
)
def test_filter_by_tag_operators(store, filter_string, expected):
    _seed_skills(store)
    with session_scope(store) as session:
        _add_skill(session, name="ml-review", organization="acme")
        _add_skill_tag(
            session, name="ml-review", organization="acme", key="team", value="platform-ml"
        )
        _add_skill(session, name="untagged-skill", organization="acme")

    with session_scope(store, commit=False) as session:
        query = apply_skill_registry_filters(
            session.query(SqlSkill),
            SearchSkillUtils.parse_search_filter(filter_string),
            _skill_column_map(),
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        assert {skill.name for skill in query.all()} == expected


_KEYWORD_TAGS = {
    "matching": {"my organization in GitHub": "yes", "API version in use": "v2"},
    "other": {"my organization in GitHub": "no", "API version in use": "v1"},
}
_ESCAPED_NEWLINE_VALUE = "a\\\nb organization = x"


def _add_tagged_registry_row(session, parent_model, tag_model, name, tags):
    ident = {"workspace": "default", "organization": "acme", "name": name}
    extra = {}
    if parent_model is SqlSkillVersion:
        session.add(SqlSkill(**ident))
        ident["version"] = 1
        extra = {"status": "active"}
    elif parent_model is SqlAgentPluginVersion:
        session.add(SqlAgentPlugin(**ident))
        ident["version"] = "1.0.0"
        extra = {"plugin_json": {"$schema": "https://example.com", "name": name}}
    session.add(parent_model(**ident, **extra))
    session.add_all(tag_model(**ident, key=key, value=value) for key, value in tags.items())


@pytest.mark.parametrize(
    ("utils", "parent_model", "tag_model"),
    [
        (SearchSkillUtils, SqlSkill, SqlSkillTag),
        (SearchSkillVersionUtils, SqlSkillVersion, SqlSkillVersionTag),
        (SearchAgentPluginUtils, SqlAgentPlugin, SqlAgentPluginTag),
        (SearchAgentPluginVersionUtils, SqlAgentPluginVersion, SqlAgentPluginVersionTag),
    ],
    ids=["skill", "skill_version", "agent_plugin", "agent_plugin_version"],
)
@pytest.mark.parametrize(
    "filter_string",
    [
        "tags.`my organization in GitHub` = 'yes'",
        "tags.`API version in use` = 'v2'",
        f"tags.note = '{_ESCAPED_NEWLINE_VALUE}' AND organization = 'acme'",
    ],
    ids=["backtick-organization", "backtick-version", "escaped-newline-value"],
)
def test_filter_matches_quoted_tag_keys_and_values_containing_keywords(
    store, utils, parent_model, tag_model, filter_string
):
    with session_scope(store) as session:
        _add_tagged_registry_row(
            session,
            parent_model,
            tag_model,
            "matching",
            {**_KEYWORD_TAGS["matching"], "note": _ESCAPED_NEWLINE_VALUE},
        )
        _add_tagged_registry_row(
            session, parent_model, tag_model, "other", {**_KEYWORD_TAGS["other"], "note": "a"}
        )

    join_keys = ["workspace", "organization", "name"]
    if hasattr(tag_model, "version"):
        join_keys.append("version")
    with session_scope(store, commit=False) as session:
        query = apply_skill_registry_filters(
            session.query(parent_model),
            utils.parse_search_filter(filter_string),
            {"organization": parent_model.organization},
            parent_model,
            tag_model,
            tag_join_keys=join_keys,
            dialect=store.engine.dialect.name,
        )
        assert [row.name for row in query.all()] == ["matching"]


# ---------------------------------------------------------------------------
# Order-by parsing
# ---------------------------------------------------------------------------


def test_order_by_valid_key():
    column_map = {"name": SqlSkill.name, "creation_timestamp": SqlSkill.created_at}
    clauses = parse_skill_registry_order_by(
        ["name ASC"],
        valid_keys={"name", "creation_timestamp"},
        column_map=column_map,
        default_tiebreakers=[SqlSkill.name.asc()],
    )
    assert len(clauses) == 1


def test_order_by_rejects_invalid_key():
    column_map = {"name": SqlSkill.name}
    with pytest.raises(MlflowException, match=r"(?i)invalid order_by key"):
        parse_skill_registry_order_by(
            ["nonexistent ASC"],
            valid_keys={"name"},
            column_map=column_map,
            default_tiebreakers=[SqlSkill.name.asc()],
        )


def test_order_by_rejects_duplicate():
    column_map = {"name": SqlSkill.name}
    with pytest.raises(MlflowException, match=r"(?i)duplicate"):
        parse_skill_registry_order_by(
            ["name ASC", "name DESC"],
            valid_keys={"name"},
            column_map=column_map,
            default_tiebreakers=[SqlSkill.name.asc()],
        )


def test_order_by_adds_default_tiebreaker():
    column_map = {"name": SqlSkill.name, "creation_timestamp": SqlSkill.created_at}
    clauses = parse_skill_registry_order_by(
        ["creation_timestamp DESC"],
        valid_keys={"name", "creation_timestamp"},
        column_map=column_map,
        default_tiebreakers=[SqlSkill.name.asc()],
    )
    assert len(clauses) == 2


def test_order_by_none_uses_defaults():
    column_map = {"name": SqlSkill.name}
    clauses = parse_skill_registry_order_by(
        None,
        valid_keys={"name"},
        column_map=column_map,
        default_tiebreakers=[SqlSkill.name.asc()],
    )
    assert len(clauses) == 1


def test_order_by_deduplicates_computed_expression_tiebreaker():
    # Each call builds a new expression object, so deduplication must compare
    # structure rather than identity or column keys.
    clauses = parse_skill_registry_order_by(
        ["status DESC"],
        valid_keys={"status"},
        column_map={"status": SqlSkill.resolved_status_expression()},
        default_tiebreakers=[
            SqlSkill.resolved_status_expression().asc(),
            SqlSkill.name.asc(),
        ],
    )
    assert len(clauses) == 2
    assert clauses[0].compare(SqlSkill.resolved_status_expression().desc())
    assert clauses[1].compare(SqlSkill.name.asc())


_SEMVER_COLUMNS = (
    SqlAgentPluginVersion.version_major,
    SqlAgentPluginVersion.version_minor,
    SqlAgentPluginVersion.version_patch,
    SqlAgentPluginVersion.version_prerelease_sort_key,
)
_VERSIONS_BY_PRECEDENCE = ["1.0.0-alpha", "1.0.0-beta", "1.0.0", "2.0.0", "9.1.0", "10.0.0"]


@pytest.mark.parametrize(
    ("order_by", "expected"),
    [
        ("version DESC", _VERSIONS_BY_PRECEDENCE[::-1]),
        ("version ASC", _VERSIONS_BY_PRECEDENCE),
    ],
)
def test_order_by_multi_column_key_sorts_semver_by_precedence(store, order_by, expected):
    with session_scope(store) as session:
        session.add(SqlAgentPlugin(workspace="default", organization="acme", name="my-plugin"))
        for version in ["2.0.0", "1.0.0", "10.0.0", "1.0.0-beta", "9.1.0", "1.0.0-alpha"]:
            session.add(
                SqlAgentPluginVersion(
                    workspace="default",
                    organization="acme",
                    name="my-plugin",
                    version=version,
                    plugin_json={"$schema": "https://example.com", "name": "my-plugin"},
                )
            )

    clauses = parse_skill_registry_order_by(
        [order_by],
        valid_keys={"version"},
        column_map={"version": _SEMVER_COLUMNS},
        default_tiebreakers=[],
    )
    with session_scope(store, commit=False) as session:
        ordered = [v.version for v in session.query(SqlAgentPluginVersion).order_by(*clauses)]
        # Precedence must match the ORM's own latest-version ordering.
        latest_first = session.query(SqlAgentPluginVersion).order_by(
            *SqlAgentPlugin._version_order_by()
        )
        assert [v.version for v in latest_first] == _VERSIONS_BY_PRECEDENCE[::-1]
    assert ordered == expected


def test_order_by_multi_column_key_deduplicates_tiebreakers():
    clauses = parse_skill_registry_order_by(
        ["version DESC"],
        valid_keys={"version"},
        column_map={"version": _SEMVER_COLUMNS},
        default_tiebreakers=[
            SqlAgentPluginVersion.version_major.desc(),
            SqlAgentPluginVersion.created_at.desc(),
        ],
    )
    assert len(clauses) == 5
    assert clauses[-1].compare(SqlAgentPluginVersion.created_at.desc())


def test_order_by_deduplicates_tiebreaker_with_mismatched_key_name():
    column_map = {"creation_timestamp": SqlSkill.created_at}
    clauses = parse_skill_registry_order_by(
        ["creation_timestamp ASC"],
        valid_keys={"creation_timestamp"},
        column_map=column_map,
        default_tiebreakers=[SqlSkill.created_at.asc()],
    )
    assert len(clauses) == 1


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_paginate_results_no_next_page():
    items = ["a", "b", "c"]
    result = paginate_results(
        items,
        max_results=5,
        offset=0,
        filter_string=None,
        order_by=None,
        query_scope="skills",
    )
    assert list(result) == ["a", "b", "c"]
    assert result.token is None


def test_paginate_results_with_next_page():
    items = ["a", "b", "c", "d"]
    result = paginate_results(
        items,
        max_results=3,
        offset=0,
        filter_string=None,
        order_by=None,
        query_scope="skills",
    )
    assert list(result) == ["a", "b", "c"]
    assert result.token is not None
    decoded = SkillRegistryPaginationToken.decode(result.token)
    assert decoded.offset == 3
    assert decoded.query_scope == "skills"


def test_paginate_results_empty():
    result = paginate_results(
        [],
        max_results=10,
        offset=0,
        filter_string=None,
        order_by=None,
        query_scope="skills",
    )
    assert list(result) == []
    assert result.token is None


# ---------------------------------------------------------------------------
# member_name filter
# ---------------------------------------------------------------------------


def test_member_name_filter(store):
    with session_scope(store) as session:
        _add_skill(session, name="code-review", organization="acme")
        session.add(
            SqlSkillVersion(
                workspace="default",
                organization="acme",
                name="code-review",
                version=1,
            )
        )
        session.add(
            SqlAgentPlugin(
                workspace="default",
                organization="acme",
                name="my-plugin",
            )
        )
        session.add(
            SqlAgentPluginVersion(
                workspace="default",
                organization="acme",
                name="my-plugin",
                version="1.0.0",
                plugin_json={"$schema": "https://example.com", "name": "my-plugin"},
            )
        )
        session.add(
            SqlAgentPluginVersionMember(
                plugin_workspace="default",
                plugin_organization="acme",
                plugin_name="my-plugin",
                plugin_version="1.0.0",
                member_organization="acme",
                member_name="code-review",
                member_version=1,
            )
        )
        session.add(
            SqlAgentPlugin(
                workspace="default",
                organization="beta",
                name="other-plugin",
            )
        )

    with session_scope(store, commit=False) as session:
        query = session.query(SqlAgentPlugin)
        query = apply_member_name_filter(
            query,
            "code-review",
            dialect=store.engine.dialect.name,
        )
        results = query.all()
        assert len(results) == 1
        assert results[0].name == "my-plugin"


# ---------------------------------------------------------------------------
# End-to-end pagination through the helpers: no duplicates, no omissions
# ---------------------------------------------------------------------------


def _page_through_skills(store, filter_string, order_by, default_tiebreakers, page_size):
    """Collect every page the way a store search method would, using only the helpers."""
    names: list[str] = []
    offset = 0
    with session_scope(store, commit=False) as session:
        while True:
            query = apply_skill_registry_filters(
                session.query(SqlSkill),
                SearchSkillUtils.parse_search_filter(filter_string),
                _skill_column_map(),
                SqlSkill,
                SqlSkillTag,
                tag_join_keys=["workspace", "organization", "name"],
                dialect=store.engine.dialect.name,
            )
            clauses = parse_skill_registry_order_by(
                order_by,
                valid_keys={"description", "name"},
                column_map={"description": SqlSkill.description, "name": SqlSkill.name},
                default_tiebreakers=default_tiebreakers,
            )
            rows = query.order_by(*clauses).offset(offset).limit(page_size + 1).all()
            page = paginate_results(
                [row.name for row in rows],
                max_results=page_size,
                offset=offset,
                filter_string=filter_string,
                order_by=order_by,
                query_scope="skills",
            )
            names.extend(page)
            if page.token is None:
                return names
            token = SkillRegistryPaginationToken.decode(page.token)
            token.validate(filter_string=filter_string, order_by=order_by, query_scope="skills")
            offset = token.offset


def test_pagination_through_helpers_returns_every_row_once(store):
    # Every acme skill ties on description, so only the tiebreaker orders them.
    # Names are inserted ascending and the tiebreaker sorts descending, so the
    # expected order cannot come from insertion or primary-key order.
    acme_names = [f"skill-{i}" for i in range(7)]
    with session_scope(store) as session:
        for name in acme_names:
            _add_skill(session, name=name, organization="acme", description="shared")
        for name in ["skill-8", "skill-9"]:
            _add_skill(session, name=name, organization="beta", description="shared")

    names = _page_through_skills(
        store,
        filter_string="organization = 'acme'",
        order_by=["description ASC"],
        default_tiebreakers=[SqlSkill.name.desc()],
        page_size=3,
    )

    assert names == sorted(acme_names, reverse=True)


# ---------------------------------------------------------------------------
# member_name excludes deleted/withdrawn versions
# ---------------------------------------------------------------------------


def test_member_name_excludes_deleted_skill_version(store):
    with session_scope(store) as session:
        _add_skill(session, name="code-review", organization="acme")
        sv = SqlSkillVersion(
            workspace="default",
            organization="acme",
            name="code-review",
            version=1,
            status="active",
        )
        session.add(sv)
        session.add(
            SqlAgentPlugin(
                workspace="default",
                organization="acme",
                name="my-plugin",
            )
        )
        session.add(
            SqlAgentPluginVersion(
                workspace="default",
                organization="acme",
                name="my-plugin",
                version="1.0.0",
                plugin_json={"$schema": "https://example.com", "name": "my-plugin"},
            )
        )
        session.add(
            SqlAgentPluginVersionMember(
                plugin_workspace="default",
                plugin_organization="acme",
                plugin_name="my-plugin",
                plugin_version="1.0.0",
                member_organization="acme",
                member_name="code-review",
                member_version=1,
            )
        )

    with session_scope(store, commit=False) as session:
        query = session.query(SqlAgentPlugin)
        results = apply_member_name_filter(
            query,
            "code-review",
            dialect=store.engine.dialect.name,
        ).all()
        assert len(results) == 1

    # Deleting the member skill version makes the plugin version withdrawn.
    with session_scope(store) as session:
        sv = session.get(
            SqlSkillVersion,
            ("default", "acme", "code-review", 1),
        )
        sv.status = "deleted"

    with session_scope(store, commit=False) as session:
        query = session.query(SqlAgentPlugin)
        results = apply_member_name_filter(
            query,
            "code-review",
            dialect=store.engine.dialect.name,
        ).all()
        assert results == []


def test_member_name_excludes_deleted_plugin_version(store):
    with session_scope(store) as session:
        _add_skill(session, name="code-review", organization="acme")
        session.add(
            SqlSkillVersion(
                workspace="default",
                organization="acme",
                name="code-review",
                version=1,
                status="active",
            )
        )
        session.add(
            SqlAgentPlugin(
                workspace="default",
                organization="acme",
                name="my-plugin",
            )
        )
        session.add(
            SqlAgentPluginVersion(
                workspace="default",
                organization="acme",
                name="my-plugin",
                version="1.0.0",
                status="deleted",
                plugin_json={"$schema": "https://example.com", "name": "my-plugin"},
            )
        )
        session.add(
            SqlAgentPluginVersionMember(
                plugin_workspace="default",
                plugin_organization="acme",
                plugin_name="my-plugin",
                plugin_version="1.0.0",
                member_organization="acme",
                member_name="code-review",
                member_version=1,
            )
        )

    with session_scope(store, commit=False) as session:
        query = session.query(SqlAgentPlugin)
        results = apply_member_name_filter(
            query,
            "code-review",
            dialect=store.engine.dialect.name,
        ).all()
        assert results == []


# ---------------------------------------------------------------------------
# Order-by with sqlparse keyword fields
# ---------------------------------------------------------------------------


def test_order_by_organization_keyword():
    column_map = {
        "organization": SqlSkill.organization,
        "name": SqlSkill.name,
    }
    clauses = parse_skill_registry_order_by(
        ["organization DESC"],
        valid_keys={"organization", "name"},
        column_map=column_map,
        default_tiebreakers=[SqlSkill.name.asc()],
    )
    assert len(clauses) == 2


@pytest.mark.parametrize("field", ["name", "organization", "version"])
@pytest.mark.parametrize("quote", ["", "`", '"'])
@pytest.mark.parametrize("direction", ["", "ASC", "DESC"])
def test_order_by_accepts_plain_and_quoted_identifiers(field, quote, direction):
    # organization and version are sqlparse keywords; name is a control.
    column = getattr(SqlSkillVersion, field)
    identifier = f"{quote}{field}{quote}"
    order_by = f"{identifier} {direction}" if direction else identifier
    clauses = parse_skill_registry_order_by(
        [order_by],
        valid_keys={field},
        column_map={field: column},
        default_tiebreakers=[],
    )
    expected = column.desc() if direction == "DESC" else column.asc()
    assert len(clauses) == 1
    assert clauses[0].compare(expected)


@pytest.mark.parametrize("field", ["name", "organization", "version"])
@pytest.mark.parametrize("direction", ["", "ASC", "DESC"])
def test_order_by_rejects_single_quoted_identifiers(field, direction):
    order_by = f"'{field}' {direction}".strip()
    with pytest.raises(MlflowException, match=r"(?i)order_by") as exc:
        parse_skill_registry_order_by(
            [order_by],
            valid_keys={field},
            column_map={field: getattr(SqlSkillVersion, field)},
            default_tiebreakers=[],
        )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


# ---------------------------------------------------------------------------
# Deleted sibling member withdraws entire version from member search
# ---------------------------------------------------------------------------


def test_member_name_excludes_version_with_deleted_sibling(store):
    with session_scope(store) as session:
        for name, status in [("code-review", "active"), ("unsafe-helper", "deleted")]:
            _add_skill(session, name=name, organization="acme")
            session.add(
                SqlSkillVersion(
                    workspace="default",
                    organization="acme",
                    name=name,
                    version=1,
                    status=status,
                )
            )
        session.add(
            SqlAgentPlugin(
                workspace="default",
                organization="acme",
                name="my-plugin",
            )
        )
        session.add(
            SqlAgentPluginVersion(
                workspace="default",
                organization="acme",
                name="my-plugin",
                version="1.0.0",
                plugin_json={"$schema": "https://example.com", "name": "my-plugin"},
            )
        )
        for name in ("code-review", "unsafe-helper"):
            session.add(
                SqlAgentPluginVersionMember(
                    plugin_workspace="default",
                    plugin_organization="acme",
                    plugin_name="my-plugin",
                    plugin_version="1.0.0",
                    member_organization="acme",
                    member_name=name,
                    member_version=1,
                )
            )

    with session_scope(store, commit=False) as session:
        query = session.query(SqlAgentPlugin)
        results = apply_member_name_filter(
            query,
            "code-review",
            dialect=store.engine.dialect.name,
        ).all()
        assert results == []


def _add_plugin_version_with_members(session, version, members, workspace="default"):
    session.add(
        SqlAgentPluginVersion(
            workspace=workspace,
            organization="acme",
            name="my-plugin",
            version=version,
            plugin_json={"$schema": "https://example.com", "name": "my-plugin"},
        )
    )
    for member_name in members:
        session.add(
            SqlAgentPluginVersionMember(
                plugin_workspace=workspace,
                plugin_organization="acme",
                plugin_name="my-plugin",
                plugin_version=version,
                member_organization="acme",
                member_name=member_name,
                member_version=1,
            )
        )


def _seed_member_skills(session, statuses, workspace="default"):
    for name, status in statuses.items():
        session.add(SqlSkill(workspace=workspace, organization="acme", name=name))
        session.add(
            SqlSkillVersion(
                workspace=workspace, organization="acme", name=name, version=1, status=status
            )
        )
    session.add(SqlAgentPlugin(workspace=workspace, organization="acme", name="my-plugin"))


def _member_matches(store, member_name):
    with session_scope(store, commit=False) as session:
        query = apply_member_name_filter(
            session.query(SqlAgentPlugin), member_name, dialect=store.engine.dialect.name
        )
        return [(plugin.workspace, plugin.name) for plugin in query.all()]


def test_member_name_keeps_version_with_deprecated_sibling(store):
    with session_scope(store) as session:
        _seed_member_skills(session, {"code-review": "active", "old-helper": "deprecated"})
        _add_plugin_version_with_members(session, "1.0.0", ["code-review", "old-helper"])

    # Only deleted members withdraw a version; deprecated ones do not.
    assert _member_matches(store, "code-review") == [("default", "my-plugin")]


def test_member_name_matches_older_eligible_version_when_newer_is_withdrawn(store):
    with session_scope(store) as session:
        _seed_member_skills(session, {"code-review": "active", "unsafe-helper": "deleted"})
        _add_plugin_version_with_members(session, "1.0.0", ["code-review"])
        _add_plugin_version_with_members(session, "2.0.0", ["code-review", "unsafe-helper"])

    assert _member_matches(store, "code-review") == [("default", "my-plugin")]


# ---------------------------------------------------------------------------
# Workspace isolation
# ---------------------------------------------------------------------------


def test_member_name_isolated_by_workspace(store):
    # Both workspaces hold the same skill and plugin identity; only team-a's
    # plugin lists the member, so team-b's plugin must not match.
    with session_scope(store) as session:
        for workspace in ("team-a", "team-b"):
            _seed_member_skills(session, {"code-review": "active"}, workspace=workspace)
        _add_plugin_version_with_members(session, "1.0.0", ["code-review"], workspace="team-a")
        _add_plugin_version_with_members(session, "1.0.0", [], workspace="team-b")

    assert _member_matches(store, "code-review") == [("team-a", "my-plugin")]


def test_tag_filter_isolated_by_workspace(store):
    # The same skill identity exists in both workspaces; only team-b's is tagged.
    with session_scope(store) as session:
        for workspace in ("team-a", "team-b"):
            _add_skill(session, name="code-review", organization="acme", workspace=workspace)
        _add_skill_tag(
            session,
            name="code-review",
            organization="acme",
            key="team",
            value="platform",
            workspace="team-b",
        )

    with session_scope(store, commit=False) as session:
        query = apply_skill_registry_filters(
            session.query(SqlSkill),
            SearchSkillUtils.parse_search_filter("tags.team = 'platform'"),
            _skill_column_map(),
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        assert [(skill.workspace, skill.name) for skill in query.all()] == [
            ("team-b", "code-review")
        ]


# ---------------------------------------------------------------------------
# Tag comparator validation
# ---------------------------------------------------------------------------


def test_unsupported_tag_comparator_rejected():
    with Session() as session, pytest.raises(MlflowException, match=r"(?i)invalid comparator"):
        apply_skill_registry_filters(
            session.query(SqlSkill),
            SearchSkillUtils.parse_search_filter("tags.team > 'a'"),
            {},
            SqlSkill,
            SqlSkillTag,
            ["workspace", "organization", "name"],
            "sqlite",
        )


def test_unsupported_tag_comparator_rejected_for_hand_built_filters():
    # Filter dicts built without the parser are still checked before reaching SQL.
    hand_built = [{"type": "tag", "key": "team", "comparator": ">", "value": "a"}]
    with Session() as session, pytest.raises(MlflowException, match=r"(?i)invalid comparator"):
        apply_skill_registry_filters(
            session.query(SqlSkill),
            hand_built,
            {},
            SqlSkill,
            SqlSkillTag,
            ["workspace", "organization", "name"],
            "sqlite",
        )


# ---------------------------------------------------------------------------
# Timestamp filters against real rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filter_string", "expected"),
    [
        ("created_at > 2000", {"newer"}),
        ("created_at >= 2000", {"newer"}),
        ("created_at < 2000", {"older"}),
        ("created_at = 1000", {"older"}),
        ("created_at != 1000", {"newer"}),
        ("last_updated_at >= 3000", {"newer"}),
    ],
)
def test_filter_by_timestamp_columns(store, filter_string, expected):
    # Parsing binds numeric values as integers, so they compare against the
    # BigInteger columns without relying on the engine to coerce a string.
    with session_scope(store) as session:
        for name, stamp in [("older", 1000), ("newer", 3000)]:
            skill = _add_skill(session, name=name, organization="acme")
            skill.created_at = stamp
            skill.last_updated_at = stamp

    with session_scope(store, commit=False) as session:
        query = apply_skill_registry_filters(
            session.query(SqlSkill),
            SearchSkillUtils.parse_search_filter(filter_string),
            {"created_at": SqlSkill.created_at, "last_updated_at": SqlSkill.last_updated_at},
            SqlSkill,
            SqlSkillTag,
            tag_join_keys=["workspace", "organization", "name"],
            dialect=store.engine.dialect.name,
        )
        assert {row.name for row in query.all()} == expected
