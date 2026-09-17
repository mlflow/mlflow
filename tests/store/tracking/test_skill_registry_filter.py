# Integration tests for the shared skill registry filter, order-by, and
# pagination helpers.  Uses a migrated SQLite database (via ``db_uri``) with
# seed data to verify that filters, tag joins, ordering, and pagination
# produce correct results against a real SQL engine.

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import (
    SqlAgentPlugin,
    SqlAgentPluginVersion,
    SqlAgentPluginVersionMember,
    SqlSkill,
    SqlSkillTag,
    SqlSkillVersion,
)
from mlflow.store.tracking.skill_registry_filter import (
    apply_member_name_filter,
    apply_skill_registry_filters,
    paginate_results,
    parse_skill_registry_order_by,
)
from mlflow.store.tracking.skill_registry_pagination import (
    SkillRegistryPaginationToken,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.search_utils import SearchSkillUtils


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
        yield session
        if commit:
            session.commit()


def _add_skill(session, *, name, organization="", description=None, search_text=None):
    skill = SqlSkill(
        workspace="default",
        organization=organization,
        name=name,
        description=description,
        search_text=search_text,
    )
    session.add(skill)
    return skill


def _add_skill_tag(session, *, name, organization="", key, value):
    tag = SqlSkillTag(
        workspace="default",
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
# End-to-end DB pagination: no duplicates, no omissions
# ---------------------------------------------------------------------------


def test_pagination_no_duplicates_or_omissions(store):
    with session_scope(store) as session:
        for i in range(5):
            _add_skill(session, name=f"skill-{i:02d}", organization="acme")

    all_names: list[str] = []
    offset = 0
    page_size = 2
    query_scope = "skills"

    with session_scope(store, commit=False) as session:
        # Page through all results
        for _ in range(10):  # safety bound
            query = (
                session
                .query(SqlSkill)
                .filter(SqlSkill.organization == "acme")
                .order_by(SqlSkill.name.asc())
                .offset(offset)
                .limit(page_size + 1)
            )
            rows = query.all()
            page = paginate_results(
                [r.name for r in rows],
                max_results=page_size,
                offset=offset,
                filter_string="organization = 'acme'",
                order_by=["name ASC"],
                query_scope=query_scope,
            )
            all_names.extend(page)
            if page.token is None:
                break
            decoded = SkillRegistryPaginationToken.decode(page.token)
            decoded.validate(
                filter_string="organization = 'acme'",
                order_by=["name ASC"],
                query_scope=query_scope,
            )
            offset = decoded.offset

    assert all_names == [f"skill-{i:02d}" for i in range(5)]
    assert len(all_names) == len(set(all_names))


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
