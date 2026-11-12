import json
import uuid
from unittest.mock import Mock

import pytest
from sqlalchemy import event
from sqlalchemy.dialects import mssql, mysql, postgresql

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import SqlScorer, SqlScorerVersion
from mlflow.store.tracking.scorer_list_scope import ScorerListScope
from mlflow.utils.workspace_context import WorkspaceContext

pytestmark = pytest.mark.notrackingurimock


def test_auth_scorer_names_metadata_only_lifecycle_workspace_and_chunks(store, monkeypatch):
    e = store.create_experiment("auth-names")
    scorers = [store.register_scorer(e, name, "{}") for name in ("a/b", "%2F", "é", "a ", "empty")]
    store.register_scorer(e, "a/b", "{}")
    store.delete_scorer(e, "empty", version=1)
    with WorkspaceContext("foreign"):
        foreign_e = store.create_experiment("foreign")
        foreign = store.register_scorer(foreign_e, "a/b", "{}")
    expected = {s.scorer_id: s.scorer_name for s in scorers[:-1]}
    if not store.supports_workspaces:
        expected[foreign.scorer_id] = "a/b"
    ids = [s.scorer_id for s in scorers] + [foreign.scorer_id, "missing", scorers[0].scorer_id]
    monkeypatch.setattr(store, "_ID_CHUNK_SIZE", 2)
    resolver = Mock(side_effect=AssertionError("auth lookup must not resolve payloads"))
    monkeypatch.setattr(store, "_batch_resolve_endpoint_in_serialized_scorers", resolver)
    statements = []

    def before_execute(conn, clause, multiparams, params, options):
        if clause.is_select:
            statements.append(clause)

    def forbidden_load(*args):
        pytest.fail("metadata lookup loaded scorer/version ORM objects")

    event.listen(store.engine, "before_execute", before_execute)
    event.listen(SqlScorerVersion, "load", forbidden_load)
    event.listen(SqlScorer, "load", forbidden_load)
    try:
        assert store._get_scorer_names_by_ids([]) == {}
        assert statements == []
        assert store._get_scorer_names_by_ids(ids) == expected
        assert len(statements) == 4  # 7 unique IDs, at most two ID binds per chunk
    finally:
        event.remove(store.engine, "before_execute", before_execute)
        event.remove(SqlScorerVersion, "load", forbidden_load)
        event.remove(SqlScorer, "load", forbidden_load)
    store.delete_experiment(e)
    assert store._get_scorer_names_by_ids(ids) == expected
    resolver.assert_not_called()
    for statement in statements:
        for dialect in (postgresql.dialect, mysql.dialect, mssql.dialect):
            compiled = statement.compile(
                dialect=dialect(), compile_kwargs={"render_postcompile": True}
            )
            sql = str(compiled)
            assert "serialized_scorer" not in sql
            assert "lifecycle_stage" not in sql
            assert "EXISTS" in sql
            assert len(compiled.params) <= 3  # IDs plus workspace


@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("kind", ["empty", "experiments", "pairs", "mixed", "broad", "deny_all"])
def test_scope_before_version_load_and_resolution(store, monkeypatch, singular, kind):
    e = store.create_experiment("scope")
    other = store.create_experiment("other")
    for exp in (e, other):
        for name in ("a", "a/b", "A", "é", "a "):
            for version in (1, 2):
                store.register_scorer(exp, name, json.dumps({"identity": [exp, name, version]}))
    scopes = {
        "empty": ScorerListScope(),
        "experiments": ScorerListScope(experiment_ids={e}),
        "pairs": ScorerListScope(scorer_pairs={(e, "a/b"), (other, "a")}),
        "mixed": ScorerListScope(
            experiment_ids={e},
            scorer_pairs={(e, "a"), (other, "a")},
            excluded_scorer_pairs={(e, "a")},
        ),
        "broad": ScorerListScope(
            experiment_ids=None, excluded_experiment_ids={other}, excluded_scorer_pairs={(e, "a/b")}
        ),
        "deny_all": ScorerListScope(experiment_ids=None, excluded_experiment_ids={e, other}),
    }
    scope = scopes[kind]
    loaded = []

    def on_load(row, context):
        loaded.append(json.loads(row.serialized_scorer)["identity"])

    resolver = Mock(side_effect=lambda values: values)
    monkeypatch.setattr(store, "_batch_resolve_endpoint_in_serialized_scorers", resolver)
    event.listen(SqlScorerVersion, "load", on_load)
    try:
        rows = (
            store.list_scorers(e, scorer_scope=scope)
            if singular
            else store.list_scorers_across_experiments([e, other], scorer_scope=scope)
        )
    finally:
        event.remove(SqlScorerVersion, "load", on_load)
    expected = sorted(
        (exp, name)
        for exp in ([e] if singular else [e, other])
        for name in ("a", "a/b", "A", "é", "a ")
        if scope.permits(exp, name)
    )
    assert [(r.experiment_id, r.scorer_name) for r in rows] == expected
    assert sorted((exp, name) for exp, name, _ in loaded) == expected
    assert all(v == 2 for _, _, v in loaded)
    if expected:
        assert (
            sorted(tuple(json.loads(s)["identity"][:2]) for s in resolver.call_args.args[0])
            == expected
        )
    else:
        resolver.assert_not_called()


@pytest.mark.parametrize("negative", [False, True])
def test_coarse_collation_still_checks_exact_raw_identity(store, monkeypatch, negative):
    e = store.create_experiment("coarse")
    for name in ("a", "A", "a ", "á"):
        store.register_scorer(e, name, json.dumps({"name": name}))
    scope = (
        ScorerListScope(experiment_ids=None, excluded_scorer_pairs={(e, "a")})
        if negative
        else ScorerListScope(scorer_pairs={(e, "a")})
    )
    original = store._scorer_identity_page
    # Simulate coarse SQL preselection, retaining the real complete membership check.
    monkeypatch.setattr(
        store,
        "_scorer_identity_page",
        lambda session, selector, after: original(session, True, after),
    )
    rows = store.list_scorers(e, scorer_scope=scope)
    assert {r.scorer_name for r in rows} == ({"A", "a ", "á"} if negative else {"a"})


def test_keyset_pages_large_acl_complete_exclusions_and_bind_budget(store, monkeypatch):
    e = store.create_experiment("large")
    # Bulk metadata/payload setup keeps the >1201 vector cheap and uses migrated fixtures.
    names = [f"n{i:04}" for i in range(1203)]
    with store.ManagedSessionMaker(read_only=False) as session:
        for name in names:
            sid = str(uuid.uuid4())
            session.add(SqlScorer(scorer_id=sid, experiment_id=int(e), scorer_name=name))
            session.add(
                SqlScorerVersion(
                    scorer_id=sid,
                    scorer_version=1,
                    serialized_scorer=json.dumps({"name": name}),
                    creation_time=1,
                )
            )
    monkeypatch.setattr(store, "_ID_CHUNK_SIZE", 30)
    monkeypatch.setattr(store, "_SCORER_METADATA_PAGE_SIZE", 17)
    pages = []
    original = store._scorer_identity_page

    def page(*args):
        rows = original(*args)
        pages.append(len(rows))
        return rows

    monkeypatch.setattr(store, "_scorer_identity_page", page)
    binds = []

    def before_execute(conn, cursor, statement, parameters, context, many):
        if statement.lstrip().upper().startswith("SELECT"):
            binds.append(len(parameters))

    scope = ScorerListScope(
        scorer_pairs={(e, n) for n in names}, excluded_scorer_pairs={(e, n) for n in names[::2]}
    )
    event.listen(store.engine, "before_cursor_execute", before_execute)
    try:
        rows = store.list_scorers_across_experiments([e, e], scorer_scope=scope)
    finally:
        event.remove(store.engine, "before_cursor_execute", before_execute)
    assert [r.scorer_name for r in rows] == names[1::2]
    assert max(pages) <= 17
    assert 0 in pages
    assert max(binds) <= 36  # 30 selector + workspace/lifecycle/keyset/limit/offset
    assert len(pages) > 80
    assert len(binds) == len(pages) + (len(rows) + 29) // 30
    # One whole experiment must itself drain through many bounded pages.
    pages.clear()
    rows = store.list_scorers(
        e,
        scorer_scope=ScorerListScope(
            experiment_ids=None, excluded_scorer_pairs=scope.excluded_scorer_pairs
        ),
    )
    assert [r.scorer_name for r in rows] == names[1::2]
    assert max(pages) == 17
    assert pages[-1] == 0
    assert len(pages) > 70


def test_stored_sql_unrepresentable_ids_and_caller_errors(store):
    e = store.create_experiment("valid")
    store.register_scorer(e, "a", "{}")
    bad = {"opaque-id", "9" * 64, str(2**63)}
    scope = ScorerListScope(experiment_ids={e, *bad}, scorer_pairs={(b, "a") for b in bad})
    assert [r.scorer_name for r in store.list_scorers(e, scorer_scope=scope)] == ["a"]
    assert store._filter_valid_scorer_experiment_ids([e, *bad, "9" * 10000]) == [e]
    with pytest.raises(MlflowException, match="Invalid experiment ID"):
        store.list_scorers_across_experiments(["invalid"], scorer_scope=ScorerListScope())


def test_direct_batch_active_and_missing_experiments(store):
    e = store.create_experiment("deleted")
    store.register_scorer(e, "a", "{}")
    store.delete_experiment(e)
    scope = ScorerListScope(experiment_ids=None)
    assert store.list_scorers_across_experiments([e, "999999"], scorer_scope=scope) == []
    with pytest.raises(MlflowException, match="active"):
        store.list_scorers(e, scorer_scope=scope)
    with pytest.raises(MlflowException, match="No Experiment"):
        store.list_scorers("999999", scorer_scope=scope)


@pytest.mark.parametrize("dialect", [postgresql.dialect, mysql.dialect, mssql.dialect])
def test_actual_scorer_query_dialect_compilation_and_numeric_binds(store, dialect):
    e = store.create_experiment("compile")
    scorer = store.register_scorer(e, "a/b", "{}")
    statements = []

    def before_execute(conn, clause, multiparams, params, options):
        if clause.is_select:
            statements.append(clause)

    event.listen(store.engine, "before_execute", before_execute)
    try:
        assert (
            len(
                store.list_scorers_across_experiments(
                    [e], scorer_scope=ScorerListScope(scorer_pairs={(e, "a/b")})
                )
            )
            == 1
        )
    finally:
        event.remove(store.engine, "before_execute", before_execute)
    assert len(statements) == 3  # metadata page, terminal empty page, payload query
    for statement in statements:
        compiled = statement.compile(dialect=dialect(), compile_kwargs={"render_postcompile": True})
        sql = str(compiled)
        assert "experiments.lifecycle_stage" in sql
        assert len(compiled.params) < 20
        if "serialized_scorer" in sql:
            assert scorer.scorer_id in compiled.params.values()
        else:
            # psycopg3 needs an integer binding, not a string experiment ID.
            assert any(type(value) is int and value == int(e) for value in compiled.params.values())
