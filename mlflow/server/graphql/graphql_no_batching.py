from typing import NamedTuple

from graphql.error import GraphQLError
from graphql.execution import ExecutionResult
from graphql.language.ast import DocumentNode, FieldNode

from mlflow.environment_variables import (
    MLFLOW_SERVER_GRAPHQL_MAX_ALIASES,
    MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS,
)

_MAX_DEPTH = 10
_MAX_SELECTIONS = 1000


class QueryInfo(NamedTuple):
    root_fields: int
    max_aliases: int


def scan_query(ast_node: DocumentNode) -> QueryInfo:
    """
    Scan a GraphQL query and return its information.
    """
    root_fields = 0
    max_aliases = 0
    total_selections = 0

    for definition in ast_node.definitions:
        if selection_set := getattr(definition, "selection_set", None):
            stack = [(selection_set, 1)]
            while stack:
                selection_set, depth = stack.pop()

                # check current level depth
                if depth > _MAX_DEPTH:
                    raise GraphQLError(f"Query exceeds maximum depth of {_MAX_DEPTH}")

                selections = getattr(selection_set, "selections", [])

                # check current level aliases
                current_aliases = 0
                for selection in selections:
                    if isinstance(selection, FieldNode):
                        if depth == 1:
                            root_fields += 1
                        if selection.alias:
                            current_aliases += 1
                        if selection.selection_set:
                            stack.append((selection.selection_set, depth + 1))
                        total_selections += 1
                        if total_selections > _MAX_SELECTIONS:
                            raise GraphQLError(
                                f"Query exceeds maximum total selections of {_MAX_SELECTIONS}"
                            )
                max_aliases = max(max_aliases, current_aliases)

    return QueryInfo(root_fields, max_aliases)


def check_query_safety(ast_node: DocumentNode) -> ExecutionResult | None:
    try:
        query_info = scan_query(ast_node)
    except GraphQLError as e:
        return ExecutionResult(
            data=None,
            errors=[e],
        )

    if query_info.root_fields > MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS.get():
        msg = "root fields"
        env_var = MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS
        value = query_info.root_fields
    elif query_info.max_aliases > MLFLOW_SERVER_GRAPHQL_MAX_ALIASES.get():
        msg = "aliases"
        env_var = MLFLOW_SERVER_GRAPHQL_MAX_ALIASES
        value = query_info.max_aliases
    else:
        return None
    return ExecutionResult(
        data=None,
        errors=[
            GraphQLError(
                f"GraphQL queries should have at most {env_var.get()} {msg}, "
                f"got {value} {msg}. To increase the limit, set the "
                f"{env_var.name} environment variable."
            )
        ],
    )
