import pytest
from graphql import parse
from graphql.error import GraphQLError

from mlflow.server.graphql.graphql_no_batching import check_query_safety, scan_query


def test_scan_query_root_fields():
    query = """
    {
        experiment { id }
        run { id }
    }
    """
    ast = parse(query)
    info = scan_query(ast)
    assert info.root_fields == 2


def test_scan_query_aliases():
    query = """
    {
        exp1: experiment { id }
        exp2: experiment { id }
    }
    """
    ast = parse(query)
    info = scan_query(ast)
    assert info.max_aliases == 2


def test_scan_query_inline_fragment_root_fields():
    query = """
    {
        experiment { id }
        ... on Query {
            run { id }
            metric { id }
        }
    }
    """
    ast = parse(query)
    info = scan_query(ast)
    # Inline fragments should count their fields as root fields:
    # experiment (1) + run (1) + metric (1) = 3 root fields
    assert info.root_fields == 3


def test_scan_query_inline_fragment_aliases():
    query = """
    {
        ... on Query {
            exp1: experiment { id }
            exp2: experiment { id }
        }
    }
    """
    ast = parse(query)
    info = scan_query(ast)
    # Both fields in the inline fragment use aliases
    assert info.max_aliases == 2


def test_scan_query_nested_inline_fragment():
    query = """
    {
        experiment {
            ... on Experiment {
                id
                name
            }
        }
    }
    """
    ast = parse(query)
    info = scan_query(ast)
    # Top level: 1 root field (experiment)
    assert info.root_fields == 1


@pytest.mark.parametrize(
    ("query", "expected_root_fields"),
    [
        # Direct field
        ("{ experiment { id } }", 1),
        # Inline fragment with single field
        ("{ ... on Query { experiment { id } } }", 1),
        # Multiple inline fragments
        ("{ ... on Query { run { id } } ... on Query { metric { id } } }", 2),
        # Mix of direct and inline fragment
        ("{ experiment { id } ... on Query { run { id } } }", 2),
    ],
)
def test_scan_query_root_fields_parametrized(query: str, expected_root_fields: int):
    ast = parse(query)
    info = scan_query(ast)
    assert info.root_fields == expected_root_fields


def test_check_query_safety_inline_fragment_exceeds_root_fields(monkeypatch):
    # Set max root fields to 2
    monkeypatch.setenv("MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS", "2")

    query = """
    {
        ... on Query {
            experiment { id }
            run { id }
            metric { id }
        }
    }
    """
    ast = parse(query)
    result = check_query_safety(ast)

    # Should return an error because 3 root fields exceed limit of 2
    assert result is not None
    assert len(result.errors) == 1
    assert "root fields" in result.errors[0].message


def test_check_query_safety_inline_fragment_exceeds_aliases(monkeypatch):
    # Set max aliases to 1
    monkeypatch.setenv("MLFLOW_SERVER_GRAPHQL_MAX_ALIASES", "1")

    query = """
    {
        ... on Query {
            exp1: experiment { id }
            exp2: experiment { id }
        }
    }
    """
    ast = parse(query)
    result = check_query_safety(ast)

    # Should return an error because 2 aliases exceed limit of 1
    assert result is not None
    assert len(result.errors) == 1
    assert "aliases" in result.errors[0].message


def test_scan_query_exceeds_max_depth():
    # Build a query with depth > 10
    query = "{ a { b { c { d { e { f { g { h { i { j { k { l } } } } } } } } } } }"
    ast = parse(query)

    with pytest.raises(GraphQLError, match="exceeds maximum depth"):
        scan_query(ast)


def test_scan_query_exceeds_max_selections():
    # Build a query with many fields
    fields = " ".join([f"field{i} {{ id }}" for i in range(1001)])
    query = f"{{ {fields} }}"
    ast = parse(query)

    with pytest.raises(GraphQLError, match="exceeds maximum total selections"):
        scan_query(ast)


def test_scan_query_fragment_spread_root_fields():
    query = """
    fragment QueryFields on Query {
        experiment { id }
        run { id }
        metric { id }
    }

    {
        ...QueryFields
    }
    """
    ast = parse(query)
    info = scan_query(ast)
    # Named fragment spread should count its fields as root fields
    assert info.root_fields == 3


def test_scan_query_unused_fragment_not_counted():
    query = """
    fragment UnusedFields on Query {
        experiment { id }
        run { id }
        metric { id }
    }

    {
        run { id }
    }
    """
    ast = parse(query)
    info = scan_query(ast)
    # Unused fragment should not be counted, only direct field
    assert info.root_fields == 1


def test_scan_query_fragment_spread_aliases():
    query = """
    fragment QueryFields on Query {
        exp1: experiment { id }
        exp2: experiment { id }
    }

    {
        ...QueryFields
    }
    """
    ast = parse(query)
    info = scan_query(ast)
    # Fragment spread should count its aliases
    assert info.max_aliases == 2


def test_check_query_safety_fragment_spread_exceeds_root_fields(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS", "2")

    query = """
    fragment QueryFields on Query {
        experiment { id }
        run { id }
        metric { id }
    }

    {
        ...QueryFields
    }
    """
    ast = parse(query)
    result = check_query_safety(ast)

    # Should return an error because 3 root fields from fragment exceed limit of 2
    assert result is not None
    assert len(result.errors) == 1
    assert "root fields" in result.errors[0].message


def test_check_query_safety_fragment_spread_exceeds_aliases(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_GRAPHQL_MAX_ALIASES", "1")

    query = """
    fragment QueryFields on Query {
        exp1: experiment { id }
        exp2: experiment { id }
    }

    {
        ...QueryFields
    }
    """
    ast = parse(query)
    result = check_query_safety(ast)

    # Should return an error because 2 aliases from fragment exceed limit of 1
    assert result is not None
    assert len(result.errors) == 1
    assert "aliases" in result.errors[0].message


def test_scan_query_nested_fragment_spread():
    query = """
    fragment Inner on Experiment {
        id
        name
    }

    fragment Outer on Query {
        experiment {
            ...Inner
        }
    }

    {
        ...Outer
    }
    """
    ast = parse(query)
    info = scan_query(ast)
    # Top-level fragment spread Outer has 1 field (experiment) at depth 1
    assert info.root_fields == 1


def test_scan_query_circular_fragment_reference():
    query = """
    fragment A on Query {
        ...B
    }

    fragment B on Query {
        ...A
    }

    {
        ...A
    }
    """
    ast = parse(query)
    # Should not crash or enter infinite loop with circular fragment references
    info = scan_query(ast)
    assert info.root_fields == 0
